use ash::{vk, Device};
use ash::khr::swapchain;
use swapchain::Device as SwapchainLoader;
use crate::vulkan_context::VulkanContext;

pub struct Renderer {
    pub swapchain_loader: SwapchainLoader,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub extent: vk::Extent2D,
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
}

impl Renderer {
    pub fn new(context: &VulkanContext, width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let swapchain_loader = swapchain::Device::new(&context.instance, &context.device);
        
        let surface_capabilities = unsafe {
            context.surface_loader.get_physical_device_surface_capabilities(context.physical_device, context.surface)?
        };
        
        let extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D { width, height }
        };

        let surface_formats = unsafe {
            context.surface_loader.get_physical_device_surface_formats(context.physical_device, context.surface)?
        };
        let format = surface_formats.first().unwrap_or(&vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(context.surface)
            .min_image_count(surface_capabilities.min_image_count + 1)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        
        let image_views: Vec<vk::ImageView> = images.iter().map(|&image| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { context.device.create_image_view(&create_info, None).unwrap() }
        }).collect();

        // Render Pass
        let color_attachment = vk::AttachmentDescription::default()
            .format(format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass));

        let render_pass = unsafe { context.device.create_render_pass(&render_pass_info, None)? };

        let framebuffers: Vec<vk::Framebuffer> = image_views.iter().map(|&view| {
            let attachments = [view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { context.device.create_framebuffer(&create_info, None).unwrap() }
        }).collect();

        // Graphics Pipeline
        let vert_source = include_str!("shaders/particle.vert");
        let frag_source = include_str!("shaders/particle.frag");
        let vert_spirv = crate::pipeline_utils::compile_shader(vert_source, "particle.vert", shaderc::ShaderKind::Vertex)?;
        let frag_spirv = crate::pipeline_utils::compile_shader(frag_source, "particle.frag", shaderc::ShaderKind::Fragment)?;
        
        let vert_module = crate::pipeline_utils::create_shader_module(&context.device, &vert_spirv)?;
        let frag_module = crate::pipeline_utils::create_shader_module(&context.device, &frag_spirv)?;

        let entry_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
        
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_name),
        ];

        let vertex_binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<crate::particles::Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let vertex_attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding_description))
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attachment));

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe { context.device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            context.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|(_, e)| e)?[0]
        };

        unsafe {
            context.device.destroy_shader_module(vert_module, None);
            context.device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            image_views,
            render_pass,
            framebuffers,
            extent,
            pipeline_layout,
            graphics_pipeline,
        })
    }

    pub fn clean(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            for &framebuffer in &self.framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }
            device.destroy_render_pass(self.render_pass, None);
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
