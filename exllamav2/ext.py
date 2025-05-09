from __future__ import annotations
import torch
from torch.utils.cpp_extension import load
import os, glob
import sys
import platform
import threading
from exllamav2.util import get_basic_progress
from exllamav2.compat import safe_move_tensor

extension_name = "exllamav2_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Since Torch 2.3.0 an annoying warning is printed every time the C++ extension is loaded, unless the
# TORCH_CUDA_ARCH_LIST variable is set. The default behavior from pytorch/torch/utils/cpp_extension.py
# is copied in the function below, but without the warning.

def maybe_set_arch_list_env():

    if os.environ.get('TORCH_CUDA_ARCH_LIST', None):
        return

    if not torch.version.cuda:
        return

    arch_list = []
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        supported_sm = [int(arch.split('_')[1])
                        for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
        if not supported_sm:
            continue
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
        # Capability of the device may be higher than what's supported by the user's
        # NVCC, causing compilation error. User's NVCC is expected to match the one
        # used to build pytorch, so we use the maximum supported capability of pytorch
        # to clamp the capability.
        capability = min(max_supported_sm, capability)
        arch = f'{capability[0]}.{capability[1]}'
        if arch not in arch_list:
            arch_list.append(arch)
    if not arch_list:
        return
    arch_list = sorted(arch_list)
    arch_list[-1] += '+PTX'

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


# Print feedback from JIT extension build

feedback_stop_event: threading.Event
feedback_thread: threading.Thread
feedback_thread_started = False

def count_object_files(directory):
    pattern = os.path.join(directory, '*.o')
    files = glob.glob(pattern)
    return len(files)

def build_feedback():
    global feedback_stop_event
    from torch.utils.cpp_extension import _get_build_directory
    build_dir = _get_build_directory(extension_name, False)
    num_sources = len(sources_)
    num_objects = count_object_files(build_dir)

    while not feedback_stop_event.is_set():
        if num_objects != num_sources: break
        feedback_stop_event.wait(1)

    progressbar = get_basic_progress()
    progressbar.start()
    task_id = progressbar.add_task("Building C++/CUDA extension", total = num_sources)

    while not feedback_stop_event.is_set():
        num_objects = count_object_files(build_dir)
        progressbar.update(task_id, completed = num_objects)
        feedback_stop_event.wait(1)

    progressbar.stop()

def start_build_feedback():
    global feedback_stop_event, feedback_thread, feedback_thread_started
    feedback_stop_event = threading.Event()
    feedback_thread = threading.Thread(target = build_feedback)
    feedback_thread.start()
    feedback_thread_started = True


def end_build_feedback():
    global feedback_thread_started
    if feedback_thread_started:
        feedback_stop_event.set()
        feedback_thread.join()


# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

build_jit = False
try:
    import exllamav2_ext
except ModuleNotFoundError:
    build_jit = True
except ImportError as e:
    if "undefined symbol" in str(e):
        print("\"undefined symbol\" error here usually means you are attempting to load a prebuilt extension wheel "
              "that was compiled against a different version of PyTorch than the one you are you using. Please verify "
              "that the versions match.")
    raise e

if build_jit:

    # Kludge to get compilation working on Windows

    if windows:

        def find_msvc():

            # Possible locations for MSVC, in order of preference

            program_files_x64 = os.environ["ProgramW6432"]
            program_files_x86 = os.environ["ProgramFiles(x86)"]

            msvc_dirs = \
            [
                a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
                for b in ["2022", "2019", "2017"]
                for a in [program_files_x64, program_files_x86]
                for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
            ]

            for msvc_dir in msvc_dirs:
                if not os.path.exists(msvc_dir): continue

                # Prefer the latest version

                versions = sorted(os.listdir(msvc_dir), reverse = True)
                for version in versions:

                    compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                    if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                        return compiler_dir

            # No path found

            return None

        import subprocess

        # Check if cl.exe is already in the path

        try:

            subprocess.check_output(["where", "/Q", "cl"])

        # If not, try to find an installation of Visual Studio and append the compiler dir to the path

        except subprocess.CalledProcessError as e:

            cl_path = find_msvc()
            if cl_path:
                if verbose:
                    print(" -- Injected compiler path:", cl_path)
                os.environ["path"] += ";" + cl_path
            else:
                print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)

    # gcc / cl.exe flags

    if windows:
        extra_cflags = ["/Ox"]
    else:
        extra_cflags = ["-Ofast"]

    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

    # nvcc flags

    extra_cuda_cflags = ["-lineinfo", "-O3"]

    if torch.version.hip:
        extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]

    # sources

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)

    sources_ = \
    [
        "ext_bindings.cpp",
        "ext_cache.cpp",
        "ext_gemm.cpp",
        "ext_hadamard.cpp",
        "ext_norm.cpp",
        "ext_qattn.cpp",
        "ext_qmatrix.cpp",
        "ext_qmlp.cpp",
        "ext_quant.cpp",
        "ext_rope.cpp",
        "ext_stloader.cpp",
        "ext_sampling.cpp",
        "ext_element.cpp",
        "ext_tp.cpp",
        "cuda/graph.cu",
        "cuda/h_add.cu",
        "cuda/h_gemm.cu",
        "cuda/lora.cu",
        "cuda/pack_tensor.cu",
        "cuda/quantize.cu",
        "cuda/q_matrix.cu",
        "cuda/q_attn.cu",
        "cuda/q_mlp.cu",
        "cuda/q_gemm.cu",
        "cuda/rms_norm.cu",
        "cuda/head_norm.cu",
        "cuda/layer_norm.cu",
        "cuda/rope.cu",
        "cuda/cache.cu",
        "cuda/util.cu",
        "cuda/softcap.cu",
        "cuda/tp.cu",
        "cuda/comp_units/kernel_select.cu",
        "cuda/comp_units/unit_gptq_1.cu",
        "cuda/comp_units/unit_gptq_2.cu",
        "cuda/comp_units/unit_gptq_3.cu",
        "cuda/comp_units/unit_exl2_1a.cu",
        "cuda/comp_units/unit_exl2_1b.cu",
        "cuda/comp_units/unit_exl2_2a.cu",
        "cuda/comp_units/unit_exl2_2b.cu",
        "cuda/comp_units/unit_exl2_3a.cu",
        "cuda/comp_units/unit_exl2_3b.cu",
        "cpp/quantize_func.cpp",
        "cpp/profiling.cpp",
        "cpp/generator.cpp",
        "cpp/sampling.cpp",
        "cpp/sampling_avx2.cpp",
    ]

    sources = [os.path.join(sources_dir, s) for s in sources_]

    # Suppress warning

    maybe_set_arch_list_env()

    # Provide build feedback if loading takes a long time, suggesting the extension is being compiled

    if not verbose:

        def load_feedback():
            print("Loading exllamav2_ext extension (JIT)...")
            start_build_feedback()

        timer = threading.Timer(1, load_feedback)
        timer.start()

    # Load extension

    try:
        exllamav2_ext = load \
        (
            name = extension_name,
            sources = sources,
            extra_include_paths = [sources_dir],
            verbose = verbose,
            extra_ldflags = extra_ldflags,
            extra_cuda_cflags = extra_cuda_cflags,
            extra_cflags = extra_cflags
        )
    finally:
        if not verbose:
            timer.cancel()
            end_build_feedback()

ext_c = exllamav2_ext


# Dummy tensor to pass to C++ extension in place of None/NULL

none_tensor = torch.empty((1, 1), device = "meta")


# Group map needed for irregular group sizes

def make_group_map_py(q_groups: torch.Tensor, num_qrows: int) -> torch.Tensor:
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2
    row = 0
    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]
    return torch.tensor(group_map, dtype = torch.short, device = q_groups.device)

def make_group_map(q_groups: torch.Tensor, num_qrows: int) -> torch.Tensor:
    group_map = ext_c.make_group_map(q_groups.cpu(), num_qrows)
    return safe_move_tensor(group_map, q_groups.device)


# Create Q matrix

def make_q_matrix(w: dict,
                  temp_dq: torch.Tensor,
                  key: str = None,
                  prescale: float = 1,
                  max_dq_rows = 0,
                  offset_qzeros: bool = False):

    # EXL2

    if "q_weight" in w:

        w["q_scale_max"] *= prescale / 256
        if "q_perm" in w: w["q_perm"] = w["q_perm"].short()
        if "q_invperm" in w: w["q_invperm"] = w["q_invperm"].short()

        if "q_group_map" not in w:
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        return ext_c.make_q_matrix(
            w["q_weight"],
            w.get("q_perm", none_tensor),
            w.get("q_invperm", none_tensor),
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            w["q_group_map"],
            none_tensor,
            none_tensor,
            none_tensor,
            w.get("bias", none_tensor),
            temp_dq,
            max_dq_rows
        )

    # GPTQ

    elif "qweight" in w:

        if prescale != 1: w["scales"] *= prescale
        if w["scales"].dtype == torch.float: w["scales"] = w["scales"].half()

        if offset_qzeros:
            w["qzeros"] -= 0b00010001000100010001000100010001

        # GPTQ with g_idx (act_order)

        if "g_idx" in w and not (w["g_idx"] == 0).all().item():

            w["q_perm"] = torch.empty((w["qweight"].shape[0] * 8,), dtype = torch.short, device = w["qweight"].device)
            w["q_invperm"] = torch.empty_like(w["q_perm"])

            return ext_c.make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                w.get("bias", none_tensor),
                temp_dq,
                max_dq_rows
            )

        # GPTQ without g_idx

        else:

            return ext_c.make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                w.get("bias", none_tensor),
                temp_dq,
                max_dq_rows
            )