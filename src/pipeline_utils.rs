use ash::vk;


pub fn create_shader_module(
    device: &ash::Device,
    code: &[u32],
) -> Result<vk::ShaderModule, vk::Result> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&create_info, None) }
}

pub fn compile_shader(
    source: &str,
    filename: &str,
    shader_kind: shaderc::ShaderKind,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let compiler = shaderc::Compiler::new().map_err(|e| Box::new(e))?;
    let artifact = compiler.compile_into_spirv(source, shader_kind, filename, "main", None)?;
    Ok(artifact.as_binary().to_vec())
}
