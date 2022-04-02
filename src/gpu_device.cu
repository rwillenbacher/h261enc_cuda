
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#include <cuda.h>

#include "h261_decl.h"

/* width of a macroblock */
#define MB_SIZE 16

#define SYNC __syncthreads()

/* constant me_gpu_t copy from host memory */
me_gpu_t __device__ __constant__ g_me_constant_gpu_device;

/* constant h261_macroblocks_t copy from host memory */
h261_macroblocks_t __device__ __constant__ g_macroblocks_constant_gpu_device;

/* texture for reference frame */
cudaArray *g_me_gpu_reference_array;
cudaArray *g_me_gpu_reference_chroma_cb_array;
cudaArray *g_me_gpu_reference_chroma_cr_array;

texture<unsigned char, 2> g_me_gpu_reference_texture;


/* texture for current frame */
cudaArray *g_me_gpu_current_array;
cudaArray *g_me_gpu_current_chroma_cb_array;
cudaArray *g_me_gpu_current_chroma_cr_array;

texture<unsigned char, 2> g_me_gpu_current_texture;


typedef struct {
	cudaEvent_t s_event_start;
	cudaEvent_t s_event_end;
	cudaStream_t s_stream;
} me_gpu_cuda_t;

typedef struct {
	cudaEvent_t s_event_start;
	cudaEvent_t s_event_end;
	cudaStream_t s_stream;
} macroblocks_cuda_t;


Void h261_gpu_init( void )
{
	Int32 i_device_count, i_device;

	cudaGetDeviceCount( &i_device_count );
	for (i_device = 0; i_device < i_device_count; i_device++ )
	{
		cudaDeviceProp s_device_prop;
		cudaGetDeviceProperties(&s_device_prop, i_device);
		
		printf("device: %s (%d)\n", s_device_prop.name, i_device );
		printf("compute caps: %d.%d\n", s_device_prop.major, s_device_prop.minor );
		printf("memory: %d\n", s_device_prop.totalGlobalMem );
		printf("processor count: %d\n", s_device_prop.multiProcessorCount );
		printf("constant memory: %d\n", s_device_prop.totalConstMem );
		printf("mem per block: %d\n", s_device_prop.sharedMemPerBlock );
		printf("memstride: %d\n", s_device_prop.memPitch );
		printf("max threads: %d\n", s_device_prop.maxThreadsPerBlock );
		printf("max threads dims: %dx%dx%d\n", s_device_prop.maxThreadsDim[ 0 ],
				s_device_prop.maxThreadsDim[ 1 ], s_device_prop.maxThreadsDim[ 2 ] );
		printf("max grid dims: %dx%dx%dx\n", s_device_prop.maxGridSize[ 0 ],
				s_device_prop.maxGridSize[ 1 ], s_device_prop.maxGridSize[ 2 ] );
	}
	
	cudaSetDevice( 0 );
	if( cudaSetDeviceFlags( cudaDeviceBlockingSync | cudaDeviceMapHost ) != cudaSuccess )
	{
		printf("cudaSetDeviceFlags failed\n");
	}
}


Void h261_gpu_device_init_textures( Int32 i_width, Int32 i_height )
{
	Int32 i_chroma_width, i_chroma_height;
	cudaChannelFormatDesc s_texture_desc;

	i_chroma_width = i_width / 2;
	i_chroma_height = i_height / 2;

	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_current_array, &s_texture_desc, i_width, i_height );
	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_current_chroma_cb_array, &s_texture_desc, i_chroma_width, i_chroma_height );
	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_current_chroma_cr_array, &s_texture_desc, i_chroma_width, i_chroma_height );

	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_reference_array, &s_texture_desc, i_width, i_height );
	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_reference_chroma_cb_array, &s_texture_desc, i_chroma_width, i_chroma_height );
	s_texture_desc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray( &g_me_gpu_reference_chroma_cr_array, &s_texture_desc, i_chroma_width, i_chroma_height );

}


Void h261_gpu_device_deinit_textures( )
{
	cudaFreeArray( g_me_gpu_current_array );
	cudaFreeArray( g_me_gpu_current_chroma_cb_array );
	cudaFreeArray( g_me_gpu_current_chroma_cr_array );
	cudaFreeArray( g_me_gpu_reference_array );
	cudaFreeArray( g_me_gpu_reference_chroma_cb_array );
	cudaFreeArray( g_me_gpu_reference_chroma_cr_array );
}

Void h261_gpu_device_set_current( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_current_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}


Void h261_gpu_device_set_current_chroma_cb( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_current_chroma_cb_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}


Void h261_gpu_device_set_current_chroma_cr( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_current_chroma_cr_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}


Void h261_gpu_device_set_reference( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_reference_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}


Void h261_gpu_device_set_reference_chroma_cb( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_reference_chroma_cb_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}


Void h261_gpu_device_set_reference_chroma_cr( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height )
{
	Int32 i_data_length;
	
	i_data_length = i_width * i_height;
	cudaMemcpyToArray( g_me_gpu_reference_chroma_cr_array, 0, 0, pui8_frame_data,
					   sizeof( UInt8 ) * i_data_length, cudaMemcpyHostToDevice );
}



/**************************************************************************
* motion estimation
**************************************************************************/

#include "megpu_kernels.cu"


Void h261_gpu_device_motion_vector_limits( me_gpu_t *ps_me_gpu )
{
	Int32 i_x, i_y, i_idx;
	Int32 i_min_x, i_min_y, i_max_x, i_max_y, i_base_x, i_base_y;
	Int32 i_global_min_x, i_global_min_y, i_global_max_x, i_global_max_y;

	i_global_min_x = 0;
	i_global_max_x = ( ps_me_gpu->i_mb_width - 1 ) * 16;
	i_global_min_y = 0;
	i_global_max_y = ( ps_me_gpu->i_mb_height - 1 ) * 16;

	for( i_y = 0; i_y < ps_me_gpu->i_mb_height; i_y++ )
	{
		for( i_x = 0; i_x < ps_me_gpu->i_mb_width; i_x++ )
		{
			i_base_x = i_x * 16;
			i_base_y = i_y * 16;
			i_min_x = i_base_x - 15;
			i_max_x = i_base_x + 15;
			i_min_y = i_base_y - 15;
			i_max_y = i_base_y + 15;
			i_min_x = MAX( i_min_x, i_global_min_x );
			i_max_x = MIN( i_max_x, i_global_max_x );
			i_min_y = MAX( i_min_y, i_global_min_y );
			i_max_y = MIN( i_max_y, i_global_max_y );

			i_idx = ( i_x + i_y * ps_me_gpu->i_mb_width ) * 4;
			ps_me_gpu->pi8_motion_vector_limits_host[ i_idx + 0 ] = i_min_x - i_base_x;
			ps_me_gpu->pi8_motion_vector_limits_host[ i_idx + 1 ] = i_max_x - i_base_x;
			ps_me_gpu->pi8_motion_vector_limits_host[ i_idx + 2 ] = i_min_y - i_base_y;
			ps_me_gpu->pi8_motion_vector_limits_host[ i_idx + 3 ] = i_max_y - i_base_y;
		}
	}
	cudaMemcpy( ps_me_gpu->pi8_motion_vector_limits_device,
				ps_me_gpu->pi8_motion_vector_limits_host,
				ps_me_gpu->i_num_mb * sizeof( Int8 ) * 4,
				cudaMemcpyHostToDevice );
				
	
}

Void h261_gpu_device_motion_vector_zero_candidates( me_gpu_t *ps_me_gpu )
{
	Int32 i_idx;
	for( i_y = 0; i_y < ps_me_gpu->i_mb_height; i_y++ )
	{
		for( i_x = 0; i_x < ps_me_gpu->i_mb_width; i_x++ )
		{
			i_idx = ( i_x + i_y * ps_me_gpu->i_mb_width );
			ps_me_gpu->ps_candidate_vector_host[ i_idx ].i8_mx = 0;
			ps_me_gpu->ps_candidate_vector_host[ i_idx ].i8_my = 0;
		}
	}
	cudaMemcpy( ps_me_gpu->ps_candidate_vector_device,
				ps_me_gpu->ps_candidate_vector_host,
				ps_me_gpu->i_num_mb * sizeof( me_gpu_mv_t ),
				cudaMemcpyHostToDevice );
}


Void h261_gpu_device_init_me( me_gpu_t **pps_me_gpu, Int32 i_width, Int32 i_height )
{
	me_gpu_t *ps_me_gpu;
	me_gpu_cuda_t *ps_me_gpu_cuda;

	cudaMallocHost( &ps_me_gpu, sizeof( me_gpu_t ) );
	ps_me_gpu->i_mb_width = i_width / MB_SIZE;
	ps_me_gpu->i_mb_height = i_height / MB_SIZE;
	ps_me_gpu->i_num_mb = ps_me_gpu->i_mb_width * ps_me_gpu->i_mb_height;
	
	cudaMalloc( ( Void ** ) &ps_me_gpu->prgs_macroblocks_device, sizeof( me_gpu_mb_t ) * ps_me_gpu->i_num_mb );
	cudaMallocHost( ( Void **) &ps_me_gpu->prgs_macroblocks_result, sizeof( me_gpu_mb_t ) * ps_me_gpu->i_num_mb );
	
	cudaMalloc( ( Void ** ) &ps_me_gpu->ps_candidate_vector_device, sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb );
	cudaMallocHost( ( Void ** ) &ps_me_gpu->ps_candidate_vector_host, sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb );

	cudaMalloc( ( Void ** ) &ps_me_gpu->ps_starting_vector_device, sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb );
	cudaMallocHost( ( Void ** ) &ps_me_gpu->ps_starting_vector_host, sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb );

	cudaMalloc( ( Void ** )	&ps_me_gpu->pi8_motion_vector_limits_device, sizeof( Int8 ) * ps_me_gpu->i_num_mb * 4 );
	cudaMallocHost( ( Void ** ) &ps_me_gpu->pi8_motion_vector_limits_host, sizeof( Int8 ) * ps_me_gpu->i_num_mb * 4 );

	/* cuda specific fields */
	ps_me_gpu_cuda = ( me_gpu_cuda_t * )malloc( sizeof( me_gpu_cuda_t ) );
	memset( ps_me_gpu_cuda, 0, sizeof( me_gpu_cuda_t ) );
	
	cudaEventCreateWithFlags( &ps_me_gpu_cuda->s_event_start, cudaEventBlockingSync );
	cudaEventCreateWithFlags( &ps_me_gpu_cuda->s_event_end, cudaEventBlockingSync );
	cudaStreamCreate( &ps_me_gpu_cuda->s_stream );
	
	ps_me_gpu->p_cuda = ps_me_gpu_cuda;

	h261_gpu_device_motion_vector_limits( ps_me_gpu );
	h261_gpu_device_motion_vector_zero_candidates( ps_me_gpu );

	*pps_me_gpu = ps_me_gpu;
}

Void h261_gpu_device_deinit_me( me_gpu_t **pps_me_gpu )
{
	me_gpu_t *ps_me_gpu;
	me_gpu_cuda_t *ps_me_gpu_cuda;
	
	ps_me_gpu = *pps_me_gpu;
	ps_me_gpu_cuda = ( me_gpu_cuda_t * )ps_me_gpu->p_cuda;

	cudaFree( ps_me_gpu->prgs_macroblocks_device );
	cudaFreeHost( ps_me_gpu->prgs_macroblocks_result );

	cudaFree( ps_me_gpu->ps_candidate_vector_device );
	cudaFreeHost( ps_me_gpu->ps_candidate_vector_host );

	cudaFree( ps_me_gpu->ps_starting_vector_device );
	cudaFreeHost( ps_me_gpu->ps_starting_vector_host );

	cudaFree( ps_me_gpu->pi8_motion_vector_limits_device );
	cudaFreeHost( ps_me_gpu->pi8_motion_vector_limits_host );

	cudaFreeHost( ps_me_gpu );

	cudaEventDestroy( ps_me_gpu_cuda->s_event_start );
	cudaEventDestroy( ps_me_gpu_cuda->s_event_end );
	cudaStreamDestroy( ps_me_gpu_cuda->s_stream );
	
	free( ps_me_gpu_cuda );
	
	*pps_me_gpu = 0;
}


Void h261_gpu_device_me( me_gpu_t *ps_me_gpu )
{
	me_gpu_cuda_t *ps_me_gpu_cuda;
	cudaError_t t_ret;
	double d_start, d_end;
	float f_elapsed_time;
	Int32 i_block_width, i_block_height, i_grid_width, i_grid_height, i_idx;

	ps_me_gpu_cuda = ( me_gpu_cuda_t * ) ps_me_gpu->p_cuda;	

	d_start = h261_get_time();

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_array );
	
	cudaEventRecord( ps_me_gpu_cuda->s_event_start, ps_me_gpu_cuda->s_stream );

	cudaMemcpyToSymbolAsync( g_me_constant_gpu_device, ps_me_gpu, sizeof( me_gpu_t ), 0, cudaMemcpyHostToDevice, ps_me_gpu_cuda->s_stream );

	dim3 d3_grid2( ps_me_gpu->i_mb_width, ps_me_gpu->i_mb_height );
	dim3 d3_block2( STARTING_VECTOR_DIMENSIONS, STARTING_VECTOR_DIMENSIONS );
	megpu_get_starting_vector<<< d3_grid2, d3_block2, 0, ps_me_gpu_cuda->s_stream >>>( );
	
/*
	i_block_width = REFINE_STARTING_VECTOR_BLOCK_DIM_X;
	i_block_height = REFINE_STARTING_VECTOR_BLOCK_DIM_Y;
	i_grid_width = ( ps_me_gpu->i_mb_width + i_block_width - 1 ) / i_block_width;
	i_grid_height = ( ps_me_gpu->i_mb_height + i_block_height - 1 ) / i_block_height;
	dim3 d3_grid3( i_grid_width, i_grid_height );
	dim3 d3_block3( i_block_width, i_block_height );
*/

	dim3 d3_grid3( ps_me_gpu->i_mb_width, ps_me_gpu->i_mb_height );
	dim3 d3_block3( REFINE_STARTING_VECTOR_DIMENSIONS, REFINE_STARTING_VECTOR_DIMENSIONS );
	megpu_refine_starting_vector<<< d3_grid3, d3_block3, 0, ps_me_gpu_cuda->s_stream >>>( );

	i_block_width = EVALUATE_MOTION_VECTOR_BLOCK_DIM_X;
	i_block_height = EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y;
	i_grid_width = ( ( ps_me_gpu->i_mb_width * EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM ) + i_block_width - 1 ) / i_block_width;
	i_grid_height = ( ( ps_me_gpu->i_mb_height * EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM ) + i_block_height - 1 ) / i_block_height;
	dim3 d3_grid4( i_grid_width, i_grid_height );
	dim3 d3_block4( i_block_width, i_block_height );
	
	megpu_evaluate_motion_vector_inter<<< d3_grid4, d3_block4, 0, ps_me_gpu_cuda->s_stream >>>( );
	megpu_evaluate_motion_vector_inter_filter<<< d3_grid4, d3_block4, 0, ps_me_gpu_cuda->s_stream >>>( );
	megpu_evaluate_motion_vector_intra<<< d3_grid4, d3_block4, 0, ps_me_gpu_cuda->s_stream >>>( );

	cudaMemcpyAsync( ps_me_gpu->ps_starting_vector_host, ps_me_gpu->ps_starting_vector_device,
				sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb, cudaMemcpyDeviceToHost, ps_me_gpu_cuda->s_stream );

	cudaMemcpyAsync( ps_me_gpu->prgs_macroblocks_result, ps_me_gpu->prgs_macroblocks_device,
				sizeof( me_gpu_mb_t ) * ps_me_gpu->i_num_mb, cudaMemcpyDeviceToHost, ps_me_gpu_cuda->s_stream );

	cudaMemcpyAsync( ps_me_gpu->ps_candidate_vector_device, ps_me_gpu->ps_me_gpu->ps_starting_vector_device,
				sizeof( me_gpu_mv_t ) * ps_me_gpu->i_num_mb, cudaMemcpyDeviceToDevice, ps_me_gpu_cuda->s_stream );
				
	cudaEventRecord( ps_me_gpu_cuda->s_event_end, ps_me_gpu_cuda->s_stream );

	cudaEventSynchronize( ps_me_gpu_cuda->s_event_end );

	cudaEventElapsedTime( &f_elapsed_time, ps_me_gpu_cuda->s_event_start, ps_me_gpu_cuda->s_event_end );

	for( i_idx = 0; i_idx < ps_me_gpu->i_num_mb; i_idx++ )
	{
		ps_me_gpu->prgs_macroblocks_result[ i_idx ].s_16x16_mv.i8_mx = ps_me_gpu->ps_starting_vector_host[ i_idx ].i8_mx;
		ps_me_gpu->prgs_macroblocks_result[ i_idx ].s_16x16_mv.i8_my = ps_me_gpu->ps_starting_vector_host[ i_idx ].i8_my;
	}

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	d_end = h261_get_time();
	//printf("ME time: %.3f ( %.3f )\n", d_end - d_start, f_elapsed_time );

	t_ret = cudaGetLastError();
	//printf("last error: %s\n", cudaGetErrorString( t_ret ) );
}



/**************************************************************************
* macroblock coder
**************************************************************************/

#include "mbgpu_kernels.cu"

Void h261_init_macroblocks( h261_macroblocks_t **pps_mbs, Int32 i_width, Int32 i_height, Int32 i_denoise )
{
	Int32 i_idx, i_x, i_y, i_num_blocks;
	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;
	h261_dct_denoise_t *ps_denoise;
	macroblocks_cuda_t *ps_mb_cuda;

	i_width >>= 4;
	i_height >>= 4;

	cudaMallocHost( ( Void ** )&ps_mbs, sizeof( h261_macroblocks_t ) );
	ps_mbs->i_mb_width = i_width;
	ps_mbs->i_mb_height = i_height;
	ps_mbs->i_num_mb = i_width * i_height;

	/* mb coder */

	cudaMalloc( ( Void ** ) &ps_mbs->ps_macroblocks_device, sizeof( h261_macroblock_t ) * i_width * i_height );
	cudaMallocHost( ( Void ** ) &ps_mbs->ps_macroblocks, sizeof( h261_macroblock_t ) * i_width * i_height );
	memset( ps_mbs->ps_macroblocks, 0, sizeof( h261_macroblock_t ) * i_width * i_height );

	cudaMallocHost( ( Void ** ) &ps_mbs->pi_mb_types_host, sizeof( Int32 ) * i_width * i_height );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_mb_types_device, sizeof( Int32 ) * i_width * i_height );

	cudaMallocHost( ( Void ** ) &ps_mbs->pi_mb_flags_host, sizeof( Int32 ) * i_width * i_height );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_mb_flags_device, sizeof( Int32 ) * i_width * i_height );

	cudaMallocHost( ( Void ** ) &ps_mbs->pi_mb_quant_host, sizeof( Int32 ) * i_width * i_height );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_mb_quant_device, sizeof( Int32 ) * i_width * i_height );

	cudaMallocHost( ( Void ** ) &ps_mbs->pi_mb_mv_x_host, sizeof( Int32 ) * i_width * i_height );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_mb_mv_x_device, sizeof( Int32 ) * i_width * i_height );

	cudaMallocHost( ( Void ** ) &ps_mbs->pi_mb_mv_y_host, sizeof( Int32 ) * i_width * i_height );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_mb_mv_y_device, sizeof( Int32 ) * i_width * i_height );

	for( i_y = 0; i_y < i_height; i_y++ )
	{
		for( i_x = 0; i_x < i_width; i_x++ )
		{
			ps_mb = &ps_mbs->ps_macroblocks[ i_y * i_width + i_x ];
			ps_mb->i_mb_x = i_x * 16;
			ps_mb->i_mb_y = i_y * 16;
		}
	}
	cudaMemcpy( ps_mbs->ps_macroblocks_device, ps_mbs->ps_macroblocks, i_width * i_height *
		sizeof( h261_macroblock_t ), cudaMemcpyHostToDevice );

	/* denoise */
	cudaMallocHost( ( Void ** ) &ps_mbs->ps_denoise_host, sizeof( h261_dct_denoise_t ) );
	cudaMalloc( ( Void ** ) &ps_mbs->ps_denoise_device, sizeof( h261_dct_denoise_t ) );

	ps_denoise = ps_mbs->ps_denoise_host;

	ps_denoise->i_denoise = i_denoise;
	ps_denoise->i_intra_count = 0;
	ps_denoise->i_inter_count = 0;
	for( i_idx = 0; i_idx < 64; i_idx++ )
	{
		ps_denoise->rgi_intra_noise[ i_idx ] = 0;
		ps_denoise->rgi_intra_signal_offset[ i_idx ] = 0;
		ps_denoise->rgi_inter_noise[ i_idx ] = 0;
		ps_denoise->rgi_inter_signal_offset[ i_idx ] = 0;
	}
	cudaMemcpy( ps_mbs->ps_denoise_device, ps_mbs->ps_denoise_host,
		sizeof( h261_dct_denoise_t ), cudaMemcpyHostToDevice );

	/* decoder */
	cudaMalloc( ( Void ** ) &ps_mbs->pui8_reconstructed_Y_device, i_width * i_height * 256 * sizeof( UInt8 ) );
	cudaMallocHost( ( Void ** ) &ps_mbs->pui8_reconstructed_Y_host, i_width * i_height * 256 * sizeof( UInt8 ) );

	cudaMalloc( ( Void ** ) &ps_mbs->pui8_reconstructed_Cb_device, i_width * i_height * 256 * sizeof( UInt8 ) );
	cudaMallocHost( ( Void ** ) &ps_mbs->pui8_reconstructed_Cb_host, i_width * i_height * 256 * sizeof( UInt8 ) );

	cudaMalloc( ( Void ** ) &ps_mbs->pui8_reconstructed_Cr_device, i_width * i_height * 256 * sizeof( UInt8 ) );
	cudaMallocHost( ( Void ** ) &ps_mbs->pui8_reconstructed_Cr_host, i_width * i_height * 256 * sizeof( UInt8 ) );

	/* coeff blocks */
	i_num_blocks = ( ( i_width * 6 * i_height ) + ( COEFF_BLOCK_WIDTH - 1 ) ) / COEFF_BLOCK_WIDTH;
	ps_mbs->i_num_blocks = i_num_blocks;
	ps_mbs->i_num_coeff_blocks = i_num_blocks * COEFF_BLOCK_WIDTH;
	cudaMalloc( ( Void ** ) &ps_mbs->pi_coeff_blocks_device, i_num_blocks * COEFF_BLOCK_SIZE * sizeof( Int32 ) );
	cudaMalloc( ( Void ** ) &ps_mbs->pi_coeff_blocks_host, i_num_blocks * COEFF_BLOCK_SIZE * sizeof( Int32 ) );

	/* cuda specific fields */
	ps_mb_cuda = ( macroblocks_cuda_t * )malloc( sizeof( macroblocks_cuda_t ) );
	memset( ps_mb_cuda, 0, sizeof( macroblocks_cuda_t ) );
	
	cudaEventCreateWithFlags( &ps_mb_cuda->s_event_start, cudaEventBlockingSync );
	cudaEventCreateWithFlags( &ps_mb_cuda->s_event_end, cudaEventBlockingSync );
	cudaStreamCreate( &ps_mb_cuda->s_stream );
	
	ps_mbs->p_cuda = ( Void * )ps_mb_cuda;


	*pps_mbs = ps_mbs;
}


Void h261_deinit_macroblocks( h261_macroblocks_t *ps_mbs )
{
	macroblocks_cuda_t *ps_mb_cuda;
	
	ps_mb_cuda = ( macroblocks_cuda_t * )ps_mbs->p_cuda;
	
	cudaFree( ps_mbs->ps_macroblocks_device );
	cudaFreeHost( ps_mbs->ps_macroblocks );

	cudaFreeHost( ps_mbs->pi_mb_types_host );
	cudaFree( ps_mbs->pi_mb_types_device );

	cudaFreeHost( ps_mbs->pi_mb_flags_host );
	cudaFree( ps_mbs->pi_mb_flags_device );

	cudaFreeHost( ps_mbs->pi_mb_quant_host );
	cudaFree( ps_mbs->pi_mb_quant_device );

	cudaFreeHost( ps_mbs->pi_mb_mv_x_host );
	cudaFree( ps_mbs->pi_mb_mv_x_device );

	cudaFreeHost( ps_mbs->pi_mb_mv_y_host );
	cudaFree( ps_mbs->pi_mb_mv_y_device );

	cudaFreeHost( ps_mbs->ps_denoise_host );
	cudaFree( ps_mbs->ps_denoise_device );

	cudaFree( ps_mbs->pui8_reconstructed_Y_device );
	cudaFreeHost( ps_mbs->pui8_reconstructed_Y_host );

	cudaFree( ps_mbs->pui8_reconstructed_Cb_device );
	cudaFreeHost( ps_mbs->pui8_reconstructed_Cb_host );

	cudaFree( ps_mbs->pui8_reconstructed_Cr_device );
	cudaFreeHost( ps_mbs->pui8_reconstructed_Cr_host );

	cudaFree( ps_mbs->pi_coeff_blocks_device );
	cudaFreeHost( ps_mbs->pi_coeff_blocks_host );

	cudaFreeHost( ps_mbs );
	
	cudaEventDestroy( ps_mb_cuda->s_event_start );
	cudaEventDestroy( ps_mb_cuda->s_event_end );
	cudaStreamDestroy( ps_mb_cuda->s_stream );
	
	free( ps_mb_cuda );
	
	
}

Void h261_gpu_device_encode_macroblocks( h261_context_t *ps_ctx )
{
	Int32 i_x, i_y, i_num_mb, i_quantiser;
	Int32 i_block_width, i_block_height, i_grid_width, i_grid_height;
	double d_start, d_end;
	float f_elapsed_time;
	
	h261_picture_t *ps_picture;
	h261_macroblocks_t *ps_mbs;
	h261_mode_decision_t *ps_md;
	h261_mb_mode_decision_t *ps_mb_md;
	macroblocks_cuda_t *ps_mb_cuda;
	
	ps_picture = &ps_ctx->s_picture;
	ps_mbs = ps_ctx->ps_macroblocks;
	ps_md = ps_ctx->ps_mode_decision;

	ps_mb_cuda = ( macroblocks_cuda_t * )ps_mbs->p_cuda;

	d_start = h261_get_time();

	for( i_y = 0; i_y < ps_mbs->i_mb_height; i_y++ )
	{
		for( i_x = 0; i_x < ps_mbs->i_mb_width; i_x++ )
		{
			ps_mb_md = &ps_md->prgs_mb[ ( i_y * ps_md->i_mb_width ) + i_x ];

			ps_mbs->pi_mb_types_host[ ( i_y * ps_mbs->i_mb_width ) + i_x ] = ps_mb_md->i_mb_type;
			ps_mbs->pi_mb_flags_host[ ( i_y * ps_mbs->i_mb_width ) + i_x ] = ps_mb_md->i_mb_flags;
			ps_mbs->pi_mb_mv_x_host[ ( i_y * ps_mbs->i_mb_width ) + i_x ] = ps_mb_md->i_mv_x;
			ps_mbs->pi_mb_mv_y_host[ ( i_y * ps_mbs->i_mb_width ) + i_x ] = ps_mb_md->i_mv_y;

			i_quantiser = ps_picture->i_quantiser;
			i_quantiser += ps_mb_md->i_quantiser_adjust;
			ps_mbs->pi_mb_quant_host[ ( i_y * ps_mbs->i_mb_width ) + i_x ] = MIN( 31, MAX( 1, i_quantiser ) );
		}
	}

	cudaEventRecord( ps_mb_cuda->s_event_start, ps_mb_cuda->s_stream );
	cudaMemcpyToSymbolAsync( g_macroblocks_constant_gpu_device, ps_mbs, sizeof( h261_macroblocks_t ), 0, cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );

	i_num_mb = ps_mbs->i_mb_width * ps_mbs->i_mb_height;
	cudaMemcpyAsync( ps_mbs->pi_mb_types_device, ps_mbs->pi_mb_types_host, i_num_mb * sizeof( Int32 ), cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pi_mb_flags_device, ps_mbs->pi_mb_flags_host, i_num_mb * sizeof( Int32 ), cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pi_mb_quant_device, ps_mbs->pi_mb_quant_host, i_num_mb * sizeof( Int32 ), cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pi_mb_mv_x_device, ps_mbs->pi_mb_mv_x_host, i_num_mb * sizeof( Int32 ), cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pi_mb_mv_y_device, ps_mbs->pi_mb_mv_y_host, i_num_mb * sizeof( Int32 ), cudaMemcpyHostToDevice, ps_mb_cuda->s_stream );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_array );

	i_block_width = SETUP_MACROBLOCKS_LUMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid( i_grid_width, i_grid_height );
	dim3 d3_block( i_block_width, i_block_height );
	macroblocks_setup_luma<<< d3_grid, d3_block, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_chroma_cb_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_chroma_cb_array );

	i_block_width = SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid2( i_grid_width, i_grid_height );
	dim3 d3_block2( i_block_width, i_block_height );
	macroblocks_setup_chromab<<< d3_grid2, d3_block2, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_chroma_cr_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_chroma_cr_array );

	i_block_width = SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid3( i_grid_width, i_grid_height );
	dim3 d3_block3( i_block_width, i_block_height );
	macroblocks_setup_chromar<<< d3_grid3, d3_block3, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	i_block_width = DCT_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid4( i_grid_width, i_grid_height );
	dim3 d3_block4( i_block_width, i_block_height );
	macroblocks_dct_forward<<< d3_grid4, d3_block4, 0, ps_mb_cuda->s_stream >>>( );

	i_block_width = DENOISE_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid5( i_grid_width, i_grid_height );
	dim3 d3_block5( i_block_width, i_block_height );
	macroblocks_dct_denoise<<< d3_grid5, d3_block5, 0, ps_mb_cuda->s_stream >>>( );

	i_block_width = DENOISE_UPDATE_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = 1;
	i_grid_height = 1;
	dim3 d3_grid6( i_grid_width, i_grid_height );
	dim3 d3_block6( i_block_width, i_block_height );
	macroblocks_update_dct_denoise<<< d3_grid6, d3_block6, 0, ps_mb_cuda->s_stream >>>( );

	
	i_block_width = QUANT_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid7( i_grid_width, i_grid_height );
	dim3 d3_block7( i_block_width, i_block_height );
	macroblocks_quant_forward<<< d3_grid7, d3_block7, 0, ps_mb_cuda->s_stream >>>( );
	
	cudaMemcpyAsync( ps_mbs->ps_macroblocks, ps_mbs->ps_macroblocks_device, i_num_mb * sizeof( h261_macroblock_t ), cudaMemcpyDeviceToHost, ps_mb_cuda->s_stream );

	cudaEventRecord( ps_mb_cuda->s_event_end, ps_mb_cuda->s_stream );
	cudaEventSynchronize( ps_mb_cuda->s_event_end );
	
	cudaEventElapsedTime( &f_elapsed_time, ps_mb_cuda->s_event_start, ps_mb_cuda->s_event_end );

	d_end = h261_get_time();

	//printf("MB coder time: %.3fms ( %f )\n", d_end - d_start, f_elapsed_time );
}


Void h261_gpu_device_decode_macroblocks( h261_context_t *ps_ctx )
{
	Int32 i_num_mb;
	Int32 i_block_width, i_block_height, i_grid_width, i_grid_height;
	double d_start, d_end;
	float f_elapsed_time;
	
	h261_macroblocks_t *ps_mbs;
	macroblocks_cuda_t *ps_mb_cuda;

	ps_mbs = ps_ctx->ps_macroblocks;
	ps_mb_cuda = ( macroblocks_cuda_t * )ps_mbs->p_cuda;

	i_num_mb = ps_mbs->i_mb_width * ps_mbs->i_mb_height;

	d_start = h261_get_time();

	cudaEventRecord( ps_mb_cuda->s_event_start, ps_mb_cuda->s_stream );

	i_block_width = DEQUANT_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid( i_grid_width, i_grid_height );
	dim3 d3_block( i_block_width, i_block_height );
	macroblocks_quant_backward<<< d3_grid, d3_block, 0, ps_mb_cuda->s_stream >>>( );

	i_block_width = IDCT_MACROBLOCKS_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid2( i_grid_width, i_grid_height );
	dim3 d3_block2( i_block_width, i_block_height );
	macroblocks_dct_inverse<<< d3_grid2, d3_block2, 0, ps_mb_cuda->s_stream >>>( );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_array );

	i_block_width = RECON_MACROBLOCKS_LUMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid3( i_grid_width, i_grid_height );
	dim3 d3_block3( i_block_width, i_block_height );
	macroblocks_reconstruct_luma<<< d3_grid3, d3_block3, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_chroma_cb_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_chroma_cb_array );

	i_block_width = RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid4( i_grid_width, i_grid_height );
	dim3 d3_block4( i_block_width, i_block_height );
	macroblocks_reconstruct_chromab<<< d3_grid4, d3_block4, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	cudaBindTextureToArray( g_me_gpu_current_texture, g_me_gpu_current_chroma_cr_array );
	cudaBindTextureToArray( g_me_gpu_reference_texture, g_me_gpu_reference_chroma_cr_array );

	i_block_width = RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_X;
	i_block_height = 1;
	i_grid_width = ( ps_mbs->i_num_coeff_blocks + i_block_width - 1 ) / i_block_width;
	i_grid_height = 1;
	dim3 d3_grid5( i_grid_width, i_grid_height );
	dim3 d3_block5( i_block_width, i_block_height );
	macroblocks_reconstruct_chromar<<< d3_grid5, d3_block5, 0, ps_mb_cuda->s_stream >>>( );

	cudaUnbindTexture( g_me_gpu_current_texture );
	cudaUnbindTexture( g_me_gpu_reference_texture );

	cudaMemcpyAsync( ps_mbs->pui8_reconstructed_Y_host, ps_mbs->pui8_reconstructed_Y_device, i_num_mb * sizeof( UInt8 ) * 256, cudaMemcpyDeviceToHost , ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pui8_reconstructed_Cb_host, ps_mbs->pui8_reconstructed_Cb_device, i_num_mb * sizeof( UInt8 ) * 64, cudaMemcpyDeviceToHost, ps_mb_cuda->s_stream );
	cudaMemcpyAsync( ps_mbs->pui8_reconstructed_Cr_host, ps_mbs->pui8_reconstructed_Cr_device, i_num_mb * sizeof( UInt8 ) * 64, cudaMemcpyDeviceToHost, ps_mb_cuda->s_stream );

	cudaEventRecord( ps_mb_cuda->s_event_end, ps_mb_cuda->s_stream );
	cudaEventSynchronize( ps_mb_cuda->s_event_end );


	cudaEventElapsedTime( &f_elapsed_time, ps_mb_cuda->s_event_start, ps_mb_cuda->s_event_end );

	d_end = h261_get_time();

	//printf("MB decoder time: %.3fms ( %f )\n", d_end - d_start, f_elapsed_time );
}


