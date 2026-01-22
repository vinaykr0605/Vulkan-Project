use ash::{vk, Entry, Instance, Device};
use ash::khr::{surface, swapchain};
use std::ffi::CStr;
use winit::window::Window;
use winit::raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
}

impl VulkanContext {
    pub fn new(window: &Window) -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };
        
        let app_info = vk::ApplicationInfo::default()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"Vulkan Particle Demo\0") })
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"No Engine\0") })
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_1);

        let extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle()?)?;
        
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle()?,
                window.raw_window_handle()?,
                None,
            )?
        };
        
        let surface_loader = surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family_index) = unsafe {
            instance.enumerate_physical_devices()?
                .into_iter()
                .filter_map(|pdevice| {
                    instance.get_physical_device_queue_family_properties(pdevice)
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, info)| {
                            let supports_graphic_and_compute = info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
                            let supports_surface = surface_loader.get_physical_device_surface_support(pdevice, index as u32, surface).unwrap_or(false);
                            
                            if supports_graphic_and_compute && supports_surface {
                                Some((pdevice, index as u32))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .next()
                .ok_or("No suitable GPU found")?
        };

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_extensions = [swapchain::NAME.as_ptr()];
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let compute_queue = graphics_queue; // Using same queue for simplicity in this demo

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            compute_queue,
            queue_family_index,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
