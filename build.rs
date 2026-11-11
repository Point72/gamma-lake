// Leaves the Python symbols to the interpreter on macOS, for `cargo build` outside hatch-rs.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
