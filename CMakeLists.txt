cmake_minimum_required(VERSION 3.17)
project(h261_cuda)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_executable( h261cuda_app
	src/common.h
	src/gpu_device.cu
	src/gpu_device.h
	src/h261_bitstream.c
	src/h261_bitstream.h
	src/h261_decl.h
	src/h261_gob.c
	src/h261_gob.h
	src/h261_macroblock.c
	src/h261_macroblock.h
	src/h261_main.c
	src/h261_moded.c
	src/h261_moded.h
	src/h261_picture.c
	src/h261_picture.h
	src/h261_ratectrl.c
	src/h261_ratectrl.h
	src/h261_tables.c
	src/h261_tables.h
	src/h261_types.h
	src/mbgpu_kernels.cuh
	src/megpu_kernels.cuh
	src/sys_win32.c
	src/sys_win32.h
)

set_target_properties(h261cuda_app PROPERTIES CUDA_ARCHITECTURES "75")