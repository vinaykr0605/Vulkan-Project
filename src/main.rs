mod vulkan_context;
mod renderer;
mod particles;
mod pipeline_utils;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use ash::vk;
use vulkan_context::VulkanContext;
use renderer::Renderer;
use particles::ParticleSystem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Particle Demo")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?;

    let context = VulkanContext::new(&window)?;
    let mut renderer = Renderer::new(&context, 800, 600)?;
    let mut particle_system = ParticleSystem::new(&context, 10000)?;

    // Command Pool
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(context.queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { context.device.create_command_pool(&pool_info, None)? };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(renderer.framebuffers.len() as u32);
    let command_buffers = unsafe { context.device.allocate_command_buffers(&alloc_info)? };

    // Sync objects
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    
    let image_available_semaphore = unsafe { context.device.create_semaphore(&semaphore_info, None)? };
    let render_finished_semaphore = unsafe { context.device.create_semaphore(&semaphore_info, None)? };
    let in_flight_fence = unsafe { context.device.create_fence(&fence_info, None)? };

    println!("Vulkan initialized successfully! Running particle system with 10k particles.");

    event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    unsafe {
                        context.device.device_wait_idle().unwrap();
                        context.device.destroy_semaphore(image_available_semaphore, None);
                        context.device.destroy_semaphore(render_finished_semaphore, None);
                        context.device.destroy_fence(in_flight_fence, None);
                        context.device.destroy_command_pool(command_pool, None);
                        particle_system.clean(&context.device);
                        renderer.clean(&context.device);
                    }
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
                    unsafe {
                        context.device.wait_for_fences(&[in_flight_fence], true, u64::MAX).unwrap();
                        context.device.reset_fences(&[in_flight_fence]).unwrap();

                        let (image_index, _) = renderer.swapchain_loader.acquire_next_image(
                            renderer.swapchain,
                            u64::MAX,
                            image_available_semaphore,
                            vk::Fence::null(),
                        ).unwrap();

                        let cmd = command_buffers[image_index as usize];
                        context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                        
                        let begin_info = vk::CommandBufferBeginInfo::default();
                        context.device.begin_command_buffer(cmd, &begin_info).unwrap();

                        // 1. Compute Pass
                        context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, particle_system.compute_pipeline);
                        context.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            particle_system.pipeline_layout,
                            0,
                            &[particle_system.descriptor_set],
                            &[],
                        );
                        context.device.cmd_dispatch(cmd, (particle_system.count + 255) / 256, 1, 1);

                        // Barrier for buffer
                        let barrier = vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                            .buffer(particle_system.buffer)
                            .offset(0)
                            .size(vk::WHOLE_SIZE);
                        
                        context.device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::VERTEX_INPUT,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[barrier],
                            &[],
                        );

                        // 2. Graphics Pass
                        let clear_values = [vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                        }];

                        let render_pass_info = vk::RenderPassBeginInfo::default()
                            .render_pass(renderer.render_pass)
                            .framebuffer(renderer.framebuffers[image_index as usize])
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: renderer.extent,
                            })
                            .clear_values(&clear_values);

                        context.device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
                        context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, renderer.graphics_pipeline);
                        context.device.cmd_bind_vertex_buffers(cmd, 0, &[particle_system.buffer], &[0]);
                        context.device.cmd_draw(cmd, particle_system.count, 1, 0, 0);
                        context.device.cmd_end_render_pass(cmd);

                        context.device.end_command_buffer(cmd).unwrap();

                        let wait_semaphores = [image_available_semaphore];
                        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                        let signal_semaphores = [render_finished_semaphore];

                        let command_buffers_submit = [cmd];
                        let submit_info = vk::SubmitInfo::default()
                            .wait_semaphores(&wait_semaphores)
                            .wait_dst_stage_mask(&wait_stages)
                            .command_buffers(&command_buffers_submit)
                            .signal_semaphores(&signal_semaphores);

                        context.device.queue_submit(context.graphics_queue, &[submit_info], in_flight_fence).unwrap();

                        let swapchains = [renderer.swapchain];
                        let image_indices = [image_index];
                        let present_info = vk::PresentInfoKHR::default()
                            .wait_semaphores(&signal_semaphores)
                            .swapchains(&swapchains)
                            .image_indices(&image_indices);

                        renderer.swapchain_loader.queue_present(context.graphics_queue, &present_info).unwrap();
                    }
                }
                _ => (),
            },
            _ => (),
        }
    })?;



    Ok(())
}
