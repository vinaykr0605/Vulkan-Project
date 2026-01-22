use ash::vk;
use std::mem::size_of;
use bytemuck::{Pod, Zeroable};
use crate::vulkan_context::VulkanContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
}

pub struct ParticleSystem {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub count: u32,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub compute_pipeline: vk::Pipeline,
}

impl ParticleSystem {
    pub fn new(context: &VulkanContext, count: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_size = (count as usize * size_of::<Particle>()) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { context.device.create_buffer(&buffer_info, None)? };
        let mem_reqs = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        
        let mem_props = unsafe { context.instance.get_physical_device_memory_properties(context.physical_device) };
        let mem_type_index = find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, mem_props).ok_or("Failed to find memory type")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { context.device.allocate_memory(&alloc_info, None)? };
        unsafe { context.device.bind_buffer_memory(buffer, memory, 0)? };

        // Initialize particles
        let mut particles = Vec::with_capacity(count as usize);
        for _ in 0..count {
            particles.push(Particle {
                pos: [
                    (rand::random::<f32>() * 2.0 - 1.0),
                    (rand::random::<f32>() * 2.0 - 1.0),
                ],
                vel: [
                    (rand::random::<f32>() * 2.0 - 1.0) * 0.1,
                    (rand::random::<f32>() * 2.0 - 1.0) * 0.1,
                ],
            });
        }

        unsafe {
            let data_ptr = context.device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(particles.as_ptr(), data_ptr as *mut Particle, count as usize);
            context.device.unmap_memory(memory);
        }

        // Descriptors
        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&layout_binding));

        let descriptor_set_layout = unsafe { context.device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);

        let descriptor_pool = unsafe { context.device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&alloc_info)?[0] };

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(buffer_size);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe { context.device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

        // Pipeline Layout
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe { context.device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compute Pipeline
        let comp_source = include_str!("shaders/particle.comp");
        let comp_spirv = crate::pipeline_utils::compile_shader(comp_source, "particle.comp", shaderc::ShaderKind::Compute)?;
        let comp_module = crate::pipeline_utils::create_shader_module(&context.device, &comp_spirv)?;

        let entry_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(comp_module)
            .name(entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let compute_pipeline = unsafe {
            context.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|(_, e)| e)?[0]
        };

        unsafe { context.device.destroy_shader_module(comp_module, None) };

        Ok(Self {
            buffer,
            memory,
            count,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            compute_pipeline,
        })
    }

    pub fn clean(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(type_filter: u32, properties: vk::MemoryPropertyFlags, mem_props: vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0 && (mem_props.memory_types[i as usize].property_flags & properties) == properties {
            return Some(i);
        }
    }
    None
}
