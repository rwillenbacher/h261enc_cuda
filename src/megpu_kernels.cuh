

Int32 __device__ megpu_sad_16x16( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture, Int16 i_pel1_x, Int16 i_pel1_y, Int16 i_pel2_x, Int16 i_pel2_y )
{
	Int8 i_y, i_x;
	Int32 i_sad = 0;

#pragma unroll
	for( i_y = 0; i_y < MB_SIZE; i_y++ )
	{
#pragma unroll
		for( i_x = 0; i_x < MB_SIZE; i_x++ )
		{
			Int32 pel1, pel2;
			pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel1_x + i_x, (float) i_pel1_y + i_y );
			pel2 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel2_x + i_x, (float) i_pel2_y + i_y );
			i_sad = __sad( pel1, pel2, i_sad );
		}
	}
	return i_sad;
}


#define STARTING_VECTOR_DIMENSIONS	( 11 )
#define STARTING_VECTOR_NUM_THREADS ( STARTING_VECTOR_DIMENSIONS * STARTING_VECTOR_DIMENSIONS )

Void __global__ megpu_get_starting_vector( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	const Int8 rgi8_test_vectors[ STARTING_VECTOR_DIMENSIONS ] =
	{
		-14, -11, -8, -5, -2, 0, 2, 5, 8, 11, 14
	};

	__shared__ Int8 rgi8_test_vector[ STARTING_VECTOR_NUM_THREADS * 2 ];
	__shared__ UInt16 rgui16_test_vector_cost[ STARTING_VECTOR_NUM_THREADS ];

	Int32 i_block_x, i_block_y, i_block_idx, i_test_x, i_test_y, i_tid;
	Int32 i_stride, i_skip;

	Int8  *pi8_motion_vector_limits;

	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv;


	i_tid = threadIdx.x + ( threadIdx.y * STARTING_VECTOR_DIMENSIONS );

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = blockIdx.x;
	i_block_y = blockIdx.y;
	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	i_block_x *= 16;
	i_block_y *= 16;

	i_test_x = rgi8_test_vectors[ threadIdx.x ];
	i_test_y = rgi8_test_vectors[ threadIdx.y ];

	ps_mv = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];

	pi8_motion_vector_limits = ps_me_gpu->pi8_motion_vector_limits_device;
	pi8_motion_vector_limits += i_block_idx * 4;

	rgi8_test_vector[ i_tid * 2 ] = i_test_x;
	rgi8_test_vector[ i_tid * 2 + 1 ] = i_test_y;

	i_skip = 0;
	if( i_test_x < pi8_motion_vector_limits[ 0 ] )
	{
		i_skip = 1;
	}
	else if( i_test_x > pi8_motion_vector_limits[ 1 ] )
	{
		i_skip = 1;
	}
	else if( i_test_y < pi8_motion_vector_limits[ 2 ] )
	{
		i_skip = 1;
	}
	else if( i_test_y > pi8_motion_vector_limits[ 3 ] )
	{
		i_skip = 1;
	}

	if( i_skip == 0 )
	{
		rgui16_test_vector_cost[ i_tid ] = megpu_sad_16x16( ps_current_texture, ps_reference_texture, i_block_x, i_block_y, i_block_x + i_test_x, i_block_y + i_test_y );
	}
	else
	{
		rgui16_test_vector_cost[ i_tid ] = 65535;
	}

		
	for( i_stride = 1; i_stride < STARTING_VECTOR_DIMENSIONS; i_stride *= 2 )
	{
		Int8 i_best_x, i_best_y;
		UInt32 i_best_sad, i_test_sad;
		UInt32 i_test_index;

		SYNC;

		i_best_x = rgi8_test_vector[ i_tid * 2 ];
		i_best_y = rgi8_test_vector[ i_tid * 2 + 1 ];
		i_best_sad = rgui16_test_vector_cost[ i_tid ];

		i_test_x = threadIdx.x + i_stride;
		i_test_y = threadIdx.y;
		i_test_index = i_test_x + ( i_test_y * STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < STARTING_VECTOR_DIMENSIONS && i_test_y < STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}

		i_test_x = threadIdx.x;
		i_test_y = threadIdx.y + i_stride;
		i_test_index = i_test_x + ( i_test_y * STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < STARTING_VECTOR_DIMENSIONS && i_test_y < STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}

		i_test_x = threadIdx.x + i_stride;
		i_test_y = threadIdx.y + i_stride;
		i_test_index = i_test_x + ( i_test_y * STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < STARTING_VECTOR_DIMENSIONS && i_test_y < STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}

		SYNC;

		rgi8_test_vector[ i_tid * 2 ] = i_best_x;
		rgi8_test_vector[ i_tid * 2 + 1 ] = i_best_y;
		rgui16_test_vector_cost[ i_tid ] = i_best_sad;
	}

	SYNC;

	if( i_tid == 0 )
	{
		i_test_x = rgi8_test_vector[ 0 ];
		i_test_y = rgi8_test_vector[ 1 ];

		ps_mv->i8_mx = i_test_x;
		ps_mv->i8_my = i_test_y;
	}
}

/*
#define STARTING_VECTOR_DIAMOND_DIMENSIONS_X	( 4 )
#define STARTING_VECTOR_DIAMOND_DIMENSIONS_Y	( 1 )
#define STARTING_VECTOR_DIAMOND_SEARCH_THREADS_X 
#define STARTING_VECTOR_DIAMOND_SUB_THREADS_X	( 4 )
#define STARTING_VECTOR_DIAMOND_SUB_THREADS_Y	( 4 )
#define STARTING_VECTOR_DIAMOND_NUM_THREADS_X ( STARTING_VECTOR_DIAMOND_DIMENSIONS_X * STARTING_VECTOR_DIAMOND_DIMENSIONS_Y * STARTING_VECTOR_DIAMOND_SUB_THREADS )
#define STARTING_VECTOR_DIAMOND_NUM_THREADS_Y ( 1 )

Void __global__ megpu_get_starting_vector_diamond( )
{
	__shared__ UInt8 rgi8_current[ MB_SIZE * MB_SIZE ][ STARTING_VECTOR_DIAMOND_DIMENSIONS_X ];
	__shared__ UInt8 rgi8_reference[ ( MB_SIZE + 8 ) * ( MB_SIZE + 8 ) ][ STARTING_VECTOR_DIAMOND_DIMENSIONS_X ];

	Int32 i_block_x, i_block_y, i_block_idx, i_test_x, i_test_y, i_tid;
	Int32 i_min_x, i_max_x, i_min_y, i_max_y;
	Int32 i_stride, i_skip;

	Int8  *pi8_motion_vector_limits;

	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv_candidate;
	me_gpu_mv_t *ps_mv_result;

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_idx = ( blockIdx.x * STARTING_VECTOR_DIAMOND_DIMENSIONS_X ) + threadIdx.x;
	i_block_x = i_block_idx % ps_me_gpu->i_mb_width;
	i_block_y = i_block_idx / ps_me_gpu->i_mb_width;
	i_block_x *= 16;
	i_block_y *= 16;

	ps_mv_candidate = &ps_me_gpu->ps_candidate_vector_device[ i_block_idx ];
	ps_mv_result = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];

	pi8_motion_vector_limits = ps_me_gpu->pi8_motion_vector_limits_device;
	pi8_motion_vector_limits += i_block_idx * 4;

	ps_me_gpu->rgi8_test_vector[ i_tid * 2 ] = i_test_x;
	ps_me_gpu->rgi8_test_vector[ i_tid * 2 + 1 ] = i_test_y;

	i_min_x = pi8_motion_vector_limits[ 0 ];
	i_max_x = pi8_motion_vector_limits[ 1 ];
	i_min_y = pi8_motion_vector_limits[ 2 ];
	i_max_y = pi8_motion_vector_limits[ 3 ];
	

}
*/
/*
#define REFINE_STARTING_VECTOR_BLOCK_DIM_X 8
#define REFINE_STARTING_VECTOR_BLOCK_DIM_Y 6

Void __global__ megpu_refine_starting_vector( )
{
	Int32 i_block_x, i_block_y, i_block_idx;
	Int32 i_best_x, i_best_y, i_best_sad, i_iter, i_last_direction;

	Int8  *pi8_motion_vector_limits;

	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv;

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = ( blockIdx.x * REFINE_STARTING_VECTOR_BLOCK_DIM_X ) + threadIdx.x;
	i_block_y = ( blockIdx.y * REFINE_STARTING_VECTOR_BLOCK_DIM_Y ) + threadIdx.y;
	if( i_block_x >= ps_me_gpu->i_mb_width || i_block_y >= ps_me_gpu->i_mb_height )
	{
		return;
	}

	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	ps_mv = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];

	i_block_x *= 16;
	i_block_y *= 16;

	i_best_x = ps_mv->i8_mx;
	i_best_y = ps_mv->i8_my;
	i_best_sad = megpu_sad_16x16( i_block_x, i_block_y, i_block_x + i_best_x, i_block_y + i_best_y );

	pi8_motion_vector_limits = ps_me_gpu->pi8_motion_vector_limits_device;
	pi8_motion_vector_limits += i_block_idx * 4;

	i_last_direction = 0;

	for( i_iter = 0; i_iter < 20; i_iter++ )
	{
		Int32 i_direction, i_continue;
		const Int32 rgi_directions[ 4 ][ 2 ] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
		const Int32 rgi_skip_direction[ 5 ] = { -1, 1, 0, 3, 2 };

		i_continue = 0;

		for( i_direction = 0; i_direction < 4; i_direction++ )
		{
			Int32 i_mv_x, i_mv_y, i_sad;
			
			if( rgi_skip_direction[ i_last_direction ] == i_direction )
			{
				continue;
			}
			
			i_mv_x = rgi_directions[ i_direction ][ 0 ] + i_best_x;
			i_mv_y = rgi_directions[ i_direction ][ 1 ] + i_best_y;
			
			if( i_mv_x >= pi8_motion_vector_limits[ 0 ] &&
				i_mv_x <= pi8_motion_vector_limits[ 1 ] &&
				i_mv_y >= pi8_motion_vector_limits[ 2 ] &&
				i_mv_y <= pi8_motion_vector_limits[ 3 ] )
			{
				i_sad = megpu_sad_16x16( i_block_x, i_block_y, i_block_x + i_mv_x, i_block_y + i_mv_y );
				if( i_sad < i_best_sad )
				{
					i_continue = 1;
					i_last_direction = i_direction + 1;
					i_best_x = i_mv_x;
					i_best_y = i_mv_y;
					i_best_sad = i_sad;
				}
			}
		}

		if( i_continue == 0 )
		{
			break;
		}
	}
		
	ps_mv->i8_mx = i_best_x;
	ps_mv->i8_my = i_best_y;
}
*/


#define REFINE_STARTING_VECTOR_DIMENSIONS	( 7 )
#define REFINE_STARTING_VECTOR_NUM_THREADS ( REFINE_STARTING_VECTOR_DIMENSIONS * REFINE_STARTING_VECTOR_DIMENSIONS )

Void __global__ megpu_refine_starting_vector( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	const Int8 rgi8_test_vectors[ STARTING_VECTOR_DIMENSIONS ] =
	{
		-3, -2, -1, 0, 1, 2, 3
	};

	__shared__ Int8 rgi8_test_vector[ STARTING_VECTOR_NUM_THREADS * 2 ];
	__shared__ UInt16 rgui16_test_vector_cost[ STARTING_VECTOR_NUM_THREADS ];

	Int32 i_block_x, i_block_y, i_block_idx, i_test_x, i_test_y, i_tid;
	Int32 i_stride, i_skip;

	Int8  *pi8_motion_vector_limits;

	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv;


	i_tid = threadIdx.x + ( threadIdx.y * REFINE_STARTING_VECTOR_DIMENSIONS );

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = blockIdx.x;
	i_block_y = blockIdx.y;
	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	i_block_x *= 16;
	i_block_y *= 16;

	ps_mv = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];

	i_test_x = rgi8_test_vectors[ threadIdx.x ] + ps_mv->i8_mx;
	i_test_y = rgi8_test_vectors[ threadIdx.y ] + ps_mv->i8_my;

	pi8_motion_vector_limits = ps_me_gpu->pi8_motion_vector_limits_device;
	pi8_motion_vector_limits += i_block_idx * 4;

	rgi8_test_vector[ i_tid * 2 ] = i_test_x;
	rgi8_test_vector[ i_tid * 2 + 1 ] = i_test_y;

	i_skip = 0;
	if( i_test_x < pi8_motion_vector_limits[ 0 ] )
	{
		i_skip = 1;
	}
	else if( i_test_x > pi8_motion_vector_limits[ 1 ] )
	{
		i_skip = 1;
	}
	else if( i_test_y < pi8_motion_vector_limits[ 2 ] )
	{
		i_skip = 1;
	}
	else if( i_test_y > pi8_motion_vector_limits[ 3 ] )
	{
		i_skip = 1;
	}

	if( i_skip == 0 )
	{
		rgui16_test_vector_cost[ i_tid ] = megpu_sad_16x16( ps_current_texture, ps_reference_texture, i_block_x, i_block_y, i_block_x + i_test_x, i_block_y + i_test_y );
	}
	else
	{
		rgui16_test_vector_cost[ i_tid ] = 65535;
	}

		
	for( i_stride = 1; i_stride < REFINE_STARTING_VECTOR_DIMENSIONS; i_stride *= 2 )
	{
		Int8 i_best_x, i_best_y;
		UInt32 i_best_sad, i_test_sad;
		UInt32 i_test_index;

		SYNC;

		i_best_x = rgi8_test_vector[ i_tid * 2 ];
		i_best_y = rgi8_test_vector[ i_tid * 2 + 1 ];
		i_best_sad = rgui16_test_vector_cost[ i_tid ];

		i_test_x = threadIdx.x + i_stride;
		i_test_y = threadIdx.y;
		i_test_index = i_test_x + ( i_test_y * REFINE_STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < REFINE_STARTING_VECTOR_DIMENSIONS && i_test_y < REFINE_STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}

		i_test_x = threadIdx.x;
		i_test_y = threadIdx.y + i_stride;
		i_test_index = i_test_x + ( i_test_y * REFINE_STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < REFINE_STARTING_VECTOR_DIMENSIONS && i_test_y < REFINE_STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}

		i_test_x = threadIdx.x + i_stride;
		i_test_y = threadIdx.y + i_stride;
		i_test_index = i_test_x + ( i_test_y * REFINE_STARTING_VECTOR_DIMENSIONS );
		if( i_test_x < REFINE_STARTING_VECTOR_DIMENSIONS && i_test_y < REFINE_STARTING_VECTOR_DIMENSIONS )
		{
			i_test_sad = rgui16_test_vector_cost[ i_test_index ];
			if( i_test_sad < i_best_sad )
			{
				i_best_x = rgi8_test_vector[ i_test_index * 2 ];
				i_best_y = rgi8_test_vector[ i_test_index * 2 + 1 ];
				i_best_sad = i_test_sad;
			}
		}
		
		SYNC;

		rgi8_test_vector[ i_tid * 2 ] = i_best_x;
		rgi8_test_vector[ i_tid * 2 + 1 ] = i_best_y;
		rgui16_test_vector_cost[ i_tid ] = i_best_sad;
	}

	SYNC;

	if( i_tid == 0 )
	{
		i_test_x = rgi8_test_vector[ i_tid * 2 ];
		i_test_y = rgi8_test_vector[ i_tid * 2 + 1 ];

		ps_mv->i8_mx = i_test_x;
		ps_mv->i8_my = i_test_y;
	}
}



//#define SATD_8x8
#ifndef SATD_8x8
#define EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM ( 4 )
#define EVALUATE_MOTION_VECTOR_MB_X			( 2 )
#define EVALUATE_MOTION_VECTOR_MB_Y			( 2 )
#define EVALUATE_BLOCK_SIZE                 ( 4 )
#else
#define EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM ( 2 )
#define EVALUATE_MOTION_VECTOR_MB_X			( 4 )
#define EVALUATE_MOTION_VECTOR_MB_Y			( 4 )
#define EVALUATE_BLOCK_SIZE                 ( 8 )
#endif
#define EVALUATE_MOTION_VECTOR_NUM_MB		( EVALUATE_MOTION_VECTOR_MB_X * EVALUATE_MOTION_VECTOR_MB_Y )
#define EVALUATE_MOTION_VECTOR_BLOCK_DIM_X	( EVALUATE_MOTION_VECTOR_MB_X * EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM )
#define EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y	( EVALUATE_MOTION_VECTOR_MB_Y * EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM )
#define EVALUATE_MOTION_VECTOR_NUM_THREADS ( EVALUATE_MOTION_VECTOR_BLOCK_DIM_X * EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y ) 

Int32 __device__ megpu_satd_4x4_kernel( Int16 *pi16_delta )
{
	Int32 i_idx, i_satd;
	Int32 a1, a2, a3, a4;
	Int32 rgi_temp_c[ 4 ][ 4 ];
	
    i_satd = 0;

	for( i_idx = 0; i_idx < 4; i_idx++ )
	{
		a1 = pi16_delta[ 0 ] + pi16_delta[ 1 ];
		a2 = pi16_delta[ 0 ] - pi16_delta[ 1 ];
		a3 = pi16_delta[ 2 ] + pi16_delta[ 3 ];
		a4 = pi16_delta[ 2 ] - pi16_delta[ 3 ];
		rgi_temp_c[ i_idx ][ 0 ] = a1 + a3;
		rgi_temp_c[ i_idx ][ 1 ] = a2 + a4;
		rgi_temp_c[ i_idx ][ 2 ] = a1 - a3;
		rgi_temp_c[ i_idx ][ 3 ] = a2 - a4;
		pi16_delta += 4;
	}
	
	for( i_idx = 0; i_idx < 4; i_idx++ )
	{
		a1 = rgi_temp_c[ 0 ][ i_idx ] + rgi_temp_c[ 1 ][ i_idx ];
		a2 = rgi_temp_c[ 0 ][ i_idx ] - rgi_temp_c[ 1 ][ i_idx ];
		a3 = rgi_temp_c[ 2 ][ i_idx ] + rgi_temp_c[ 3 ][ i_idx ];
		a4 = rgi_temp_c[ 2 ][ i_idx ] - rgi_temp_c[ 3 ][ i_idx ];
		i_satd = __sad( 0, a1 + a3, i_satd );
		i_satd = __sad( 0, a2 + a4, i_satd );
		i_satd = __sad( a1, a3, i_satd );
		i_satd = __sad( a2, a4, i_satd );
	}
	return i_satd;
}

float __constant__ g_rgf_forward_postscale_satd[ 8 ] = { 1.0, 0.720960, 0.765367, 0.850430, 1.0, 1.272759, 1.847759, 3.624510 };


Int32 __device__ megpu_satd_8x8_kernel( Int16 *pi16_delta )
{
	float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	float tmp10, tmp11, tmp12, tmp13;
	float z1, z2, z3, z4, z5, z11, z13;
	float *pf_temp, rgf_temp[64], f_column_scale;
	Int32 i_idx;
	Int32 i_satd;

	pf_temp = &rgf_temp[ 0 ];

	for( i_idx = 0; i_idx < 8; i_idx ++ )
	{
		tmp0 = pi16_delta[ 0 ] + pi16_delta[ 7 ];
		tmp1 = pi16_delta[ 1 ] + pi16_delta[ 6 ];
		tmp2 = pi16_delta[ 2 ] + pi16_delta[ 5 ];
		tmp3 = pi16_delta[ 3 ] + pi16_delta[ 4 ];
		tmp4 = pi16_delta[ 3 ] - pi16_delta[ 4 ];
		tmp5 = pi16_delta[ 2 ] - pi16_delta[ 5 ];
		tmp6 = pi16_delta[ 1 ] - pi16_delta[ 6 ];
		tmp7 = pi16_delta[ 0 ] - pi16_delta[ 7 ];

		tmp10 = tmp0 + tmp3;
		tmp13 = tmp0 - tmp3;
		tmp11 = tmp1 + tmp2;
		tmp12 = tmp1 - tmp2;

		pf_temp[ 0 ] = tmp10 + tmp11;
		pf_temp[ 4 ] = tmp10 - tmp11;

		z1 = tmp12 + tmp13;
		z1 *= 0.707107;
		pf_temp[ 2 ]= tmp13 + z1;
		pf_temp[ 6 ]= tmp13 - z1;

		tmp4 += tmp5;
		tmp5 += tmp6;
		tmp6 += tmp7;

		z5 = (tmp4 - tmp6) * 0.382683;
		z2 = tmp4 * 0.541196 + z5;
		z4 = tmp6 * 1.306563 + z5;

		z3 = tmp5 * 0.707107;
		z11 = tmp7 + z3;
		z13 = tmp7 - z3;

		pf_temp[ 5 ] = z13 + z2;
		pf_temp[ 3 ] = z13 - z2;
		pf_temp[ 1 ] = z11 + z4;
		pf_temp[ 7 ] = z11 - z4;

		pf_temp += 8;
		pi16_delta += 8;
	}

	i_satd = 0;

	pf_temp = &rgf_temp[ 0 ];
	for( i_idx = 0; i_idx < 8; i_idx++)
	{
		f_column_scale = g_rgf_forward_postscale_satd[ i_idx ];
		f_column_scale *= 0.125;

		tmp0= pf_temp[ 0 ] + pf_temp[ 56 ];
		tmp1= pf_temp[ 8 ] + pf_temp[ 48 ];
		tmp2= pf_temp[ 16 ] + pf_temp[ 40 ];
		tmp3= pf_temp[ 24 ] + pf_temp[ 32 ];
		tmp4= pf_temp[ 24 ] - pf_temp[ 32 ];
		tmp5= pf_temp[ 16 ] - pf_temp[ 40 ];
		tmp6= pf_temp[ 8 ] - pf_temp[ 48 ];
		tmp7= pf_temp[ 0 ] - pf_temp[ 56 ];

		tmp10 = tmp0 + tmp3;
		tmp13 = tmp0 - tmp3;
		tmp11 = tmp1 + tmp2;
		tmp12 = tmp1 - tmp2;

		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 0 ] * ( tmp10 + tmp11 ) ) + 0.49999 );
		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 4 ] * ( tmp10 - tmp11 ) ) + 0.49999 ) ;

		z1 = tmp12 + tmp13;
		z1 *= 0.707107;

		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 2 ] * ( tmp13 + z1 ) ) + 0.49999 );
		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 6 ] * ( tmp13 - z1 ) ) + 0.49999 );

		tmp4 += tmp5;
		tmp5 += tmp6;
		tmp6 += tmp7;

		z5 = (tmp4 - tmp6) * 0.382683;
		z2 = tmp4 * 0.541196 + z5;
		z4 = tmp6 * 1.306563 + z5;

		z3 = tmp5 * 0.707107;
		z11 = tmp7 + z3;
		z13 = tmp7 - z3;

		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 5 ] * ( z13 + z2 ) ) + 0.49999 );
		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 3 ] * ( z13 - z2 ) ) + 0.49999 );
		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 1 ] * ( z11 + z4 ) ) + 0.49999 );
		i_satd += abs( ( f_column_scale * g_rgf_forward_postscale_satd[ 7 ] * ( z11 - z4 ) ) + 0.49999 );

		pf_temp += 1;
	}
	
	return i_satd;
}



Void __global__ megpu_evaluate_motion_vector_inter( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	Int32 i_block_x, i_block_y, i_sub_block_x, i_sub_block_y, i_block_idx, i_sub_block_idx;
	Int32 i_x, i_y, i_sub_block_mv_x, i_sub_block_mv_y;
	Int32 i_is_top_left, i_stride;
	
	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv;
	me_gpu_mb_t *ps_mb;

	Int16 rgi16_block_data[ EVALUATE_BLOCK_SIZE * EVALUATE_BLOCK_SIZE ];
	UInt32 __shared__ rgui_sub_block_satd[ EVALUATE_MOTION_VECTOR_NUM_THREADS ];

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = ( blockIdx.x * EVALUATE_MOTION_VECTOR_BLOCK_DIM_X ) + threadIdx.x;
	i_block_y = ( blockIdx.y * EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y ) + threadIdx.y;
	i_sub_block_x = i_block_x % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_sub_block_y = i_block_y % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_x = i_block_x / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_y = i_block_y / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;

	if( i_block_x >= ps_me_gpu->i_mb_width || i_block_y >= ps_me_gpu->i_mb_height )
	{
		return;
	}

	if( i_sub_block_x == 0 && i_sub_block_y == 0 )
	{
		i_is_top_left = 1;
	}
	else
	{
		i_is_top_left = 0;
	}

	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	i_sub_block_idx = ( threadIdx.y * blockDim.x ) + threadIdx.x;

	ps_mv = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];
	ps_mb = &ps_me_gpu->prgs_macroblocks_device[ i_block_idx ];

	i_sub_block_x = ( i_block_x * 16 ) + ( i_sub_block_x * EVALUATE_BLOCK_SIZE );
	i_sub_block_y = ( i_block_y * 16 ) + ( i_sub_block_y * EVALUATE_BLOCK_SIZE );

	i_sub_block_mv_x = i_sub_block_x + ps_mv->i8_mx;
	i_sub_block_mv_y = i_sub_block_y + ps_mv->i8_my;

/* inter cost */
	
	for( i_y = 0; i_y < EVALUATE_BLOCK_SIZE; i_y++ )
	{
		for( i_x = 0; i_x < EVALUATE_BLOCK_SIZE; i_x++ )
		{
			Int32 i_pel1, i_pel2;
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_sub_block_x + i_x, (float) i_sub_block_y + i_y );
			i_pel2 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y );
			rgi16_block_data[ i_x + ( i_y * EVALUATE_BLOCK_SIZE ) ] = i_pel1 - i_pel2;
		}
	}
	
#ifndef SATD_8x8
	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_4x4_kernel( &rgi16_block_data[ 0 ] );

	for( i_stride = 1; i_stride < EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM; i_stride *= 2 )
	{
		UInt32 ui_satd, ui_end;

		SYNC;

		ui_end = i_sub_block_idx + i_stride + ( i_stride * blockDim.x );
		if( ui_end < EVALUATE_MOTION_VECTOR_NUM_THREADS )
		{
			ui_satd = rgui_sub_block_satd[ i_sub_block_idx ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + ( i_stride * blockDim.x ) ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride + ( i_stride * blockDim.x ) ];
			rgui_sub_block_satd[ i_sub_block_idx ] = ui_satd;
		}
	}
#else
	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_8x8_kernel( &rgi16_block_data[ 0 ] );

	for( i_stride = 1; i_stride < EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM; i_stride *= 2 )
	{
		UInt32 ui_satd, ui_end;

		SYNC;

		ui_end = i_sub_block_idx + i_stride + ( i_stride * blockDim.x );
		if( ui_end < EVALUATE_MOTION_VECTOR_NUM_THREADS )
		{
			ui_satd = rgui_sub_block_satd[ i_sub_block_idx ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + ( i_stride * blockDim.x ) ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride + ( i_stride * blockDim.x ) ];
			rgui_sub_block_satd[ i_sub_block_idx ] = ui_satd;
		}
	}
#endif
	if( i_is_top_left )
	{
		ps_mb->i_16x16_cost = rgui_sub_block_satd[ i_sub_block_idx ];
	}
}


Void __global__ megpu_evaluate_motion_vector_inter_filter( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	Int32 i_block_x, i_block_y, i_sub_block_x, i_sub_block_y, i_block_idx, i_sub_block_idx;
	Int32 i_x, i_y, i_sub_block_mv_x, i_sub_block_mv_y;
	Int32 i_is_top_left, i_stride;
	
	Int32 i_filter_start_x, i_filter_end_x, i_filter_start_y, i_filter_end_y;

	me_gpu_t *ps_me_gpu;
	me_gpu_mv_t *ps_mv;
	me_gpu_mb_t *ps_mb;

	Int16 rgi16_block_data[ EVALUATE_BLOCK_SIZE * EVALUATE_BLOCK_SIZE ];
	UInt32 __shared__ rgui_sub_block_satd[ EVALUATE_MOTION_VECTOR_NUM_THREADS ];

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = ( blockIdx.x * EVALUATE_MOTION_VECTOR_BLOCK_DIM_X ) + threadIdx.x;
	i_block_y = ( blockIdx.y * EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y ) + threadIdx.y;
	i_sub_block_x = i_block_x % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_sub_block_y = i_block_y % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_x = i_block_x / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_y = i_block_y / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;

	if( i_block_x >= ps_me_gpu->i_mb_width || i_block_y >= ps_me_gpu->i_mb_height )
	{
		return;
	}

	if( i_sub_block_x == 0 && i_sub_block_y == 0 )
	{
		i_is_top_left = 1;
	}
	else
	{
		i_is_top_left = 0;
	}

	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	i_sub_block_idx = ( threadIdx.y * blockDim.x ) + threadIdx.x;

	ps_mv = &ps_me_gpu->ps_starting_vector_device[ i_block_idx ];
	ps_mb = &ps_me_gpu->prgs_macroblocks_device[ i_block_idx ];

	i_sub_block_x = ( i_block_x * 16 ) + ( i_sub_block_x * EVALUATE_BLOCK_SIZE );
	i_sub_block_y = ( i_block_y * 16 ) + ( i_sub_block_y * EVALUATE_BLOCK_SIZE );

	i_sub_block_mv_x = i_sub_block_x + ps_mv->i8_mx;
	i_sub_block_mv_y = i_sub_block_y + ps_mv->i8_my;

/* inter + filter cost */
	for( i_y = 0; i_y < EVALUATE_BLOCK_SIZE; i_y++ )
	{
		for( i_x = 0; i_x < EVALUATE_BLOCK_SIZE; i_x++ )
		{
			Int32 i_pel1, i_pel2;
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_sub_block_x + i_x, (float) i_sub_block_y + i_y );
			i_pel2 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y );
			rgi16_block_data[ i_x + ( i_y * EVALUATE_BLOCK_SIZE ) ] = i_pel1 - i_pel2;
		}
	}


#ifdef SATD_8x8	
	for( i_y = 1; i_y < 7; i_y++ )
	{
		for( i_x = 1; i_x < 7; i_x++ )
		{
			Int32 a1, a2, a3, a4, a5, a6, a7, a8, a9;
			Int32 i_pel1, i_pel2;
			i_pel1 = tex2D( ps_current_texture, (float) i_sub_block_x + i_x, (float) i_sub_block_y + i_y );

			a1 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y - 1 );
			a2 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y - 1 );
			a3 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y - 1 );

			a4 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y );
			a5 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y );
			a6 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y );

			a7 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y + 1 );
			a8 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y + 1 );
			a9 = tex2D( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y + 1 );

			i_pel2 = ( a1 + ( a2 * 2 ) + a3 + ( a4 * 2 ) + ( a5 * 4 ) + ( a6 * 2 ) + a7 + ( a8 * 2 ) + a9 + 8 ) / 16;

			rgi16_block_data[ i_x + ( i_y * EVALUATE_BLOCK_SIZE ) ] = i_pel1 - i_pel2;
		}
	}

	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_8x8_kernel( &rgi16_block_data[ 0 ] );

	for( i_stride = 1; i_stride < EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM; i_stride *= 2 )
	{
		UInt32 ui_satd, ui_end;

		SYNC;

		ui_end = i_sub_block_idx + i_stride + ( i_stride * blockDim.x );
		if( ui_end < EVALUATE_MOTION_VECTOR_NUM_THREADS )
		{
			ui_satd = rgui_sub_block_satd[ i_sub_block_idx ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + ( i_stride * blockDim.x ) ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride + ( i_stride * blockDim.x ) ];
			rgui_sub_block_satd[ i_sub_block_idx ] = ui_satd;
		}
	}
#else
	if( threadIdx.x & 1 )
	{
		i_filter_start_x = 0;
		i_filter_end_x = 3;
	}
	else
	{
		i_filter_start_x = 1;
		i_filter_end_x = 4;
	}

	if( threadIdx.y & 1 )
	{
		i_filter_start_y = 0;
		i_filter_end_y = 3;
	}
	else
	{
		i_filter_start_y = 1;
		i_filter_end_y = 4;
	}

	for( i_y = i_filter_start_y; i_y < i_filter_end_y; i_y++ )
	{
		for( i_x = i_filter_start_x; i_x < i_filter_end_x; i_x++ )
		{
			Int32 a1, a2, a3, a4, a5, a6, a7, a8, a9;
			Int32 i_pel1, i_pel2;
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_sub_block_x + i_x, (float) i_sub_block_y + i_y );

			a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y - 1 );
			a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y - 1 );
			a3 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y - 1 );

			a4 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y );
			a5 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y );
			a6 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y );

			a7 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x - 1, (float) i_sub_block_mv_y + i_y + 1 );
			a8 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x, (float) i_sub_block_mv_y + i_y + 1 );
			a9 = tex2D<unsigned char>( ps_reference_texture, (float) i_sub_block_mv_x + i_x + 1, (float) i_sub_block_mv_y + i_y + 1 );

			i_pel2 = ( a1 + ( a2 * 2 ) + a3 + ( a4 * 2 ) + ( a5 * 4 ) + ( a6 * 2 ) + a7 + ( a8 * 2 ) + a9 + 8 ) / 16;

			rgi16_block_data[ i_x + ( i_y * EVALUATE_BLOCK_SIZE ) ] = i_pel1 - i_pel2;
		}
	}

	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_4x4_kernel( &rgi16_block_data[ 0 ] );

	for( i_stride = 1; i_stride < EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM; i_stride *= 2 )
	{
		UInt32 ui_satd, ui_end;

		SYNC;

		ui_end = i_sub_block_idx + i_stride + ( i_stride * blockDim.x );
		if( ui_end < EVALUATE_MOTION_VECTOR_NUM_THREADS )
		{
			ui_satd = rgui_sub_block_satd[ i_sub_block_idx ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + ( i_stride * blockDim.x ) ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride + ( i_stride * blockDim.x ) ];
			rgui_sub_block_satd[ i_sub_block_idx ] = ui_satd;
		}
	}
#endif

	if( i_is_top_left )
	{
		ps_mb->i_16x16_filter_cost = rgui_sub_block_satd[ i_sub_block_idx ];
	}
}

Void __global__ megpu_evaluate_motion_vector_intra( cudaTextureObject_t ps_current_texture )
{
	Int32 i_block_x, i_block_y, i_sub_block_x, i_sub_block_y, i_block_idx, i_sub_block_idx;
	Int32 i_x, i_y;
	Int32 i_is_top_left, i_stride;
	
	me_gpu_t *ps_me_gpu;
	me_gpu_mb_t *ps_mb;

	UInt32 __shared__ rgui_sub_block_satd[ EVALUATE_MOTION_VECTOR_NUM_THREADS ];
	Int16 rgi16_block_data[ EVALUATE_BLOCK_SIZE * EVALUATE_BLOCK_SIZE ];

	ps_me_gpu = &g_me_constant_gpu_device;

	i_block_x = ( blockIdx.x * EVALUATE_MOTION_VECTOR_BLOCK_DIM_X ) + threadIdx.x;
	i_block_y = ( blockIdx.y * EVALUATE_MOTION_VECTOR_BLOCK_DIM_Y ) + threadIdx.y;
	i_sub_block_x = i_block_x % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_sub_block_y = i_block_y % EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_x = i_block_x / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;
	i_block_y = i_block_y / EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM;

	if( i_block_x >= ps_me_gpu->i_mb_width || i_block_y >= ps_me_gpu->i_mb_height )
	{
		return;
	}

	if( i_sub_block_x == 0 && i_sub_block_y == 0 )
	{
		i_is_top_left = 1;
	}
	else
	{
		i_is_top_left = 0;
	}

	i_block_idx = i_block_x + ( i_block_y * ps_me_gpu->i_mb_width );
	i_sub_block_idx = ( threadIdx.y * blockDim.x ) + threadIdx.x;

	ps_mb = &ps_me_gpu->prgs_macroblocks_device[ i_block_idx ];

	i_sub_block_x = ( i_block_x * 16 ) + ( i_sub_block_x * EVALUATE_BLOCK_SIZE );
	i_sub_block_y = ( i_block_y * 16 ) + ( i_sub_block_y * EVALUATE_BLOCK_SIZE );

	
/* intra cost */
	for( i_y = 0; i_y < EVALUATE_BLOCK_SIZE; i_y++ )
	{
		for( i_x = 0; i_x < EVALUATE_BLOCK_SIZE; i_x++ )
		{
			Int32 i_pel1;
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_sub_block_x + i_x, (float) i_sub_block_y + i_y );
			rgi16_block_data[ i_x + ( i_y * EVALUATE_BLOCK_SIZE ) ] = i_pel1;
		}
	}
#ifdef SATD_8x8
	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_8x8_kernel( &rgi16_block_data[ 0 ] );
#else
	rgui_sub_block_satd[ i_sub_block_idx ] = megpu_satd_4x4_kernel( &rgi16_block_data[ 0 ] );
#endif
	for( i_stride = 1; i_stride < EVALUATE_MOTION_VECTOR_SUB_BLOCK_DIM; i_stride *= 2 )
	{
		UInt32 ui_satd, ui_end;

		SYNC;

		ui_end = i_sub_block_idx + i_stride + ( i_stride * blockDim.x );
		if( ui_end < EVALUATE_MOTION_VECTOR_NUM_THREADS )
		{
			ui_satd = rgui_sub_block_satd[ i_sub_block_idx ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + ( i_stride * blockDim.x ) ];
			ui_satd += rgui_sub_block_satd[ i_sub_block_idx + i_stride + ( i_stride * blockDim.x ) ];
			rgui_sub_block_satd[ i_sub_block_idx ] = ui_satd;
		}
	}
	if( i_is_top_left )
	{
		ps_mb->i_intra_cost = rgui_sub_block_satd[ i_sub_block_idx ];
	}

}
