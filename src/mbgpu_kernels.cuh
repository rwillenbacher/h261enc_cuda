
// cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture

#define COEFF_SIZE 8

Void __device__ macroblocks_get_coeffs_ptr( h261_macroblocks_t *ps_mbs, Int32 i_coeff_id, Int32 **ppi_coeffs )
{
	Int32 i_block_num, i_coeff_num;
	Int32 *pi_coeffs;

	i_block_num = i_coeff_id / COEFF_BLOCK_WIDTH;
	i_coeff_num = i_coeff_id % COEFF_BLOCK_WIDTH;

	pi_coeffs = ps_mbs->pi_coeff_blocks_device;
	pi_coeffs += COEFF_BLOCK_SIZE * i_block_num;
	pi_coeffs += i_coeff_num;
	*ppi_coeffs = pi_coeffs;
}

Void __device__ macroblocks_get_macroblock_ptr( h261_macroblocks_t *ps_mbs, Int32 i_coeff_id, h261_macroblock_t **pps_mb )
{
	Int32 i_mb_idx;

	i_mb_idx = i_coeff_id / 6;

	if( i_mb_idx >= ps_mbs->i_num_mb )
	{
		*pps_mb = 0;
	}
	else
	{
		*pps_mb = &ps_mbs->ps_macroblocks_device[ i_mb_idx ];
	}
}

Int32 __device__ macroblocks_get_macroblock_idx( h261_macroblocks_t *ps_mbs, Int32 i_coeff_id )
{
	Int32 i_mb_idx;

	i_mb_idx = i_coeff_id / 6;

	if( i_mb_idx >= ps_mbs->i_num_mb )
	{
		return -1;
	}
	return i_mb_idx;
}


Int32 __device__ macroblocks_get_coeff_idx( Int32 i_coeff_id )
{
	return i_coeff_id % 6;
}



/* motion compensation */

Void __device__ macroblocks_setup_intra( cudaTextureObject_t ps_current_texture, Int32 *pi_coeffs, Int32 i_pel_x, Int32 i_pel_y )
{
	Int32 i_x, i_y;

	for( i_y = 0; i_y < COEFF_SIZE; i_y++ )
	{
		for( i_x = 0; i_x < COEFF_SIZE; i_x++ )
		{
			Int32 i_pel;
			i_pel = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + i_y );
			pi_coeffs[ ( i_y * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel;
		}
	}
}


Void __device__ macroblocks_setup_inter( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture, Int32 *pi_coeffs, Int32 i_pel_x, Int32 i_pel_y, Int32 i_mb_flags, Int32 i_mv_x, Int32 i_mv_y )
{
	Int32 i_x, i_y;

	if( i_mb_flags & H261_MB_FILTER )
	{
		Int32 i_pel1, i_pel2;
		Int32 rgi_temp[ 64 ];

		for( i_y = 0; i_y < 8; i_y++ )
		{
			Int32 i_a1, i_a2, i_a3;

			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 0 ] = i_a1 * 4;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 1, (float) i_pel_y + i_mv_y + i_y );
			i_a3 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 2, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 1 ] = i_a1 + i_a2 * 2 + i_a3;
			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 3, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 2 ] = i_a2 + i_a3 * 2 + i_a1;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 4, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 3 ] = i_a3 + i_a1 * 2 + i_a2;
			i_a3 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 5, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 4 ] = i_a1 + i_a2 * 2 + i_a3;
			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 6, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 5 ] = i_a2 + i_a3 * 2 + i_a1;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_mv_x + 7, (float) i_pel_y + i_mv_y + i_y );
			rgi_temp[ i_y * 8 + 6 ] = i_a3 + i_a1 * 2 + i_a2;
			rgi_temp[ i_y * 8 + 7 ] = i_a2 * 4;
		}
		for( i_x = 0; i_x < 8; i_x++ )
		{
			Int32 i_a1, i_a2, i_a3;

			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 0 );
			i_a1 = rgi_temp[ 8 * 0 + i_x ];
			i_pel2 = ( i_a1 + 2 ) >> 2;
			pi_coeffs[ ( 0 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a2 = rgi_temp[ 8 * 1 + i_x ];
			i_a3 = rgi_temp[ 8 * 2 + i_x ];

			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 1 );
			i_pel2 = ( i_a1 + i_a2 * 2 + i_a3 + 8 ) >> 4;
			pi_coeffs[ ( 1 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a1 = rgi_temp[ 8 * 3 + i_x ];
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 2 );
			i_pel2 = ( i_a2 + i_a3 * 2 + i_a1 + 8 ) >> 4;
			pi_coeffs[ ( 2 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a2 = rgi_temp[ 8 * 4 + i_x ];
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 3 );
			i_pel2 = ( i_a3 + i_a1 * 2 + i_a2 + 8 ) >> 4;
			pi_coeffs[ ( 3 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a3 = rgi_temp[ 8 * 5 + i_x ];
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 4 );
			i_pel2 = ( i_a1 + i_a2 * 2 + i_a3 + 8 ) >> 4;
			pi_coeffs[ ( 4 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a1 = rgi_temp[ 8 * 6 + i_x ];
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 5 );
			i_pel2 = ( i_a2 + i_a3 * 2 + i_a1 + 8 ) >> 4;
			pi_coeffs[ ( 5 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_a2 = rgi_temp[ 8 * 7 + i_x ];
			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 6 );
			i_pel2 = ( i_a3 + i_a1 * 2 + i_a2 + 8 ) >> 4;
			pi_coeffs[ ( 6 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;

			i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + 7 );
			i_pel2 = ( i_a2 + 2 ) >> 2;
			pi_coeffs[ ( 7 * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;
		}
	}
	else
	{
		for( i_y = 0; i_y < COEFF_SIZE; i_y++ )
		{
			for( i_x = 0; i_x < COEFF_SIZE; i_x++ )
			{
				Int32 i_pel1, i_pel2;

				i_pel1 = tex2D<unsigned char>( ps_current_texture, (float) i_pel_x + i_x, (float) i_pel_y + i_y );
				i_pel2 = tex2D<unsigned char>( ps_reference_texture, (float) i_pel_x + i_x + i_mv_x, (float) i_pel_y + i_y + i_mv_y );

				pi_coeffs[ ( i_y * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ] = i_pel1 - i_pel2;
			}
		}
	}
}


#define SETUP_MACROBLOCKS_LUMA_BLOCK_DIM_X 128
#define SETUP_MACROBLOCKS_LUMA_BLOCK_DIM_Y 1

Void __global__ macroblocks_setup_luma( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_x, i_coeff_y, i_coeff_idx;
	Int32 i_mb_idx, i_mb_flags, i_mb_type;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( SETUP_MACROBLOCKS_LUMA_BLOCK_DIM_X * blockIdx.x );

	i_mb_idx = macroblocks_get_macroblock_idx( ps_mbs, i_coeff_id );
	if( i_mb_idx < 0 )
	{
		return;
	}

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx > 3 )
	{
		return;
	}

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	
	i_mb_type = ps_mbs->pi_mb_types_device[ i_mb_idx ];
	i_mb_flags = ps_mbs->pi_mb_flags_device[ i_mb_idx ];
	i_mv_x = ps_mbs->pi_mb_mv_x_device[ i_mb_idx ];
	i_mv_y = ps_mbs->pi_mb_mv_y_device[ i_mb_idx ];

	if( i_coeff_idx == 0 )
	{
		ps_mb->i_macroblock_type = ps_mbs->pi_mb_types_device[ i_mb_idx ];
		ps_mb->i_macroblock_type_flags = ps_mbs->pi_mb_flags_device[ i_mb_idx ];
		ps_mb->i_macroblock_quant = ps_mbs->pi_mb_quant_device[ i_mb_idx ];
		ps_mb->rgi_mv[ 0 ] = ps_mbs->pi_mb_mv_x_device[ i_mb_idx ];
		ps_mb->rgi_mv[ 1 ] = ps_mbs->pi_mb_mv_y_device[ i_mb_idx ];
	}
	
	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_x = i_coeff_idx & 1;
	i_coeff_y = ( i_coeff_idx & 2 ) >> 1;
	i_pel_x = ps_mb->i_mb_x + ( i_coeff_x * 8 );
	i_pel_y = ps_mb->i_mb_y + ( i_coeff_y * 8 );

	switch( i_mb_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_setup_intra( ps_current_texture, pi_coeffs, i_pel_x, i_pel_y );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_setup_inter( ps_current_texture, ps_reference_texture, pi_coeffs, i_pel_x, i_pel_y, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}


#define SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_X 128
#define SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_Y 1

Void __global__ macroblocks_setup_chromab( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx, i_mb_flags;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_X * blockIdx.x );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx != 4 )
	{
		return;
	}

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_mb_flags = ps_mb->i_macroblock_type_flags;

	i_mv_x = ps_mb->rgi_mv[ 0 ] / 2;
	i_mv_y = ps_mb->rgi_mv[ 1 ] / 2;
	i_pel_x = ps_mb->i_mb_x / 2;
	i_pel_y = ps_mb->i_mb_y / 2;

	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_setup_intra( ps_current_texture, pi_coeffs, i_pel_x, i_pel_y );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_setup_inter( ps_current_texture, ps_reference_texture, pi_coeffs, i_pel_x, i_pel_y, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}

Void __global__ macroblocks_setup_chromar( cudaTextureObject_t ps_current_texture, cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx, i_mb_flags;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( SETUP_MACROBLOCKS_CHROMA_BLOCK_DIM_X * blockIdx.x );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx != 5 )
	{
		return;
	}

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_mb_flags = ps_mb->i_macroblock_type_flags;

	i_mv_x = ps_mb->rgi_mv[ 0 ] / 2;
	i_mv_y = ps_mb->rgi_mv[ 1 ] / 2;
	i_pel_x = ps_mb->i_mb_x / 2;
	i_pel_y = ps_mb->i_mb_y / 2;

	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_setup_intra( ps_current_texture, pi_coeffs, i_pel_x, i_pel_y );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_setup_inter( ps_current_texture, ps_reference_texture, pi_coeffs, i_pel_x, i_pel_y, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}


/* dct */

#define DCT_MACROBLOCKS_BLOCK_DIM_X ( 288 )
#define DCT_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

float __constant__ g_rgf_forward_postscale[ 8 ] = { 1.0, 0.720960, 0.765367, 0.850430, 1.0, 1.272759, 1.847759, 3.624510 };

Void __device__ macroblocks_dct8x8( Int32 *pi_source )
{
	float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	float tmp10, tmp11, tmp12, tmp13;
	float z1, z2, z3, z4, z5, z11, z13;
	float *pf_temp, rgf_temp[64], f_column_scale;
	Int32 i_idx, *pi_destination;

	pf_temp = &rgf_temp[ 0 ];
	pi_destination = pi_source;

	for( i_idx = 0; i_idx < 8; i_idx ++ )
	{
		tmp0 = pi_source[ 0 * COEFF_BLOCK_STRIDE_X ] + pi_source[ 7 * COEFF_BLOCK_STRIDE_X ];
		tmp1 = pi_source[ 1 * COEFF_BLOCK_STRIDE_X ] + pi_source[ 6 * COEFF_BLOCK_STRIDE_X ];
		tmp2 = pi_source[ 2 * COEFF_BLOCK_STRIDE_X ] + pi_source[ 5 * COEFF_BLOCK_STRIDE_X ];
		tmp3 = pi_source[ 3 * COEFF_BLOCK_STRIDE_X ] + pi_source[ 4 * COEFF_BLOCK_STRIDE_X ];
		tmp4 = pi_source[ 3 * COEFF_BLOCK_STRIDE_X ] - pi_source[ 4 * COEFF_BLOCK_STRIDE_X ];
		tmp5 = pi_source[ 2 * COEFF_BLOCK_STRIDE_X ] - pi_source[ 5 * COEFF_BLOCK_STRIDE_X ];
		tmp6 = pi_source[ 1 * COEFF_BLOCK_STRIDE_X ] - pi_source[ 6 * COEFF_BLOCK_STRIDE_X ];
		tmp7 = pi_source[ 0 * COEFF_BLOCK_STRIDE_X ] - pi_source[ 7 * COEFF_BLOCK_STRIDE_X ];

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
		pi_source += COEFF_BLOCK_STRIDE_Y;
	}

	pf_temp = &rgf_temp[ 0 ];
	for( i_idx = 0; i_idx < 8; i_idx++)
	{
		f_column_scale = g_rgf_forward_postscale[ i_idx ];
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

		pi_destination[ COEFF_BLOCK_STRIDE_Y * 0 ] = floorf( ( f_column_scale * g_rgf_forward_postscale[ 0 ] * ( tmp10 + tmp11 ) ) + 0.49999 );
		pi_destination[ COEFF_BLOCK_STRIDE_Y * 4 ] = floorf( ( f_column_scale * g_rgf_forward_postscale[ 4 ] * ( tmp10 - tmp11 ) ) + 0.49999 ) ;

		z1 = tmp12 + tmp13;
		z1 *= 0.707107;

		pi_destination[ COEFF_BLOCK_STRIDE_Y * 2 ] = floorf( ( f_column_scale * g_rgf_forward_postscale[ 2 ] * ( tmp13 + z1 ) ) + 0.49999 );
		pi_destination[ COEFF_BLOCK_STRIDE_Y * 6 ] = floorf( ( f_column_scale * g_rgf_forward_postscale[ 6 ] * ( tmp13 - z1 ) ) + 0.49999 );

		tmp4 += tmp5;
		tmp5 += tmp6;
		tmp6 += tmp7;

		z5 = (tmp4 - tmp6) * 0.382683;
		z2 = tmp4 * 0.541196 + z5;
		z4 = tmp6 * 1.306563 + z5;

		z3 = tmp5 * 0.707107;
		z11 = tmp7 + z3;
		z13 = tmp7 - z3;

		pi_destination[ COEFF_BLOCK_STRIDE_Y * 5 ]= floorf( ( f_column_scale * g_rgf_forward_postscale[ 5 ] * ( z13 + z2 ) ) + 0.49999 );
		pi_destination[ COEFF_BLOCK_STRIDE_Y * 3 ]= floorf( ( f_column_scale * g_rgf_forward_postscale[ 3 ] * ( z13 - z2 ) ) + 0.49999 );
		pi_destination[ COEFF_BLOCK_STRIDE_Y * 1 ]= floorf( ( f_column_scale * g_rgf_forward_postscale[ 1 ] * ( z11 + z4 ) ) + 0.49999 );
		pi_destination[ COEFF_BLOCK_STRIDE_Y * 7 ]= floorf( ( f_column_scale * g_rgf_forward_postscale[ 7 ] * ( z11 - z4 ) ) + 0.49999 );

		pf_temp += 1;
		pi_destination += COEFF_BLOCK_STRIDE_X;
	}
}


Void __global__ macroblocks_dct_forward( )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( DCT_MACROBLOCKS_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	macroblocks_dct8x8( pi_coeffs );
}


/* denoise */

/* this has to be a power of 2 and greaterequal than 64 */
#define DENOISE_MACROBLOCKS_BLOCK_DIM_X ( 128 )
#define DENOISE_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

#define DENOISE_UPDATE_MACROBLOCKS_BLOCK_DIM_X ( 64 )
#define DENOISE_UPDATE_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

Void __global__ macroblocks_update_dct_denoise( )
{
	Int32 i_idx;

	h261_macroblocks_t *ps_mbs;
	h261_dct_denoise_t *ps_denoise;

	ps_mbs = &g_macroblocks_constant_gpu_device;
	ps_denoise = ps_mbs->ps_denoise_device;
	
	i_idx = threadIdx.x;

	ps_denoise->rgi_inter_signal_offset[ i_idx ] = ( ( ps_denoise->i_inter_count * ps_denoise->i_denoise ) +
		( ps_denoise->rgi_inter_noise[ i_idx ] / 2 ) ) / ( ps_denoise->rgi_inter_noise[ i_idx ] + 1 );
	ps_denoise->rgi_intra_signal_offset[ i_idx ] = ( ( ps_denoise->i_intra_count * ps_denoise->i_denoise ) +
		( ps_denoise->rgi_intra_noise[ i_idx ] / 2 ) ) / ( ps_denoise->rgi_intra_noise[ i_idx ] + 1 );

	ps_denoise->rgi_inter_noise[ i_idx ] = 0;
	ps_denoise->rgi_intra_noise[ i_idx ] = 0;

	if( i_idx == 0 )
	{
		ps_denoise->i_inter_count = 0;
		ps_denoise->i_intra_count = 0;
	}
}


Void __global__ macroblocks_dct_denoise( )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id;

	Int32 i_idx, i_x, i_y, i_level, i_tid, i_block_idx, i_intra, i_inter, i_stride;
	Int32 __shared__ rgi_noise_accum[ 2 ][ DENOISE_MACROBLOCKS_BLOCK_DIM_X ];
	UInt16 __shared__ rgui16_noise_offset[ 2 ][ 64 ];

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;
	h261_dct_denoise_t *ps_denoise;

	ps_mbs = &g_macroblocks_constant_gpu_device;
	ps_denoise = ps_mbs->ps_denoise_device;

	i_coeff_id = threadIdx.x + ( DENOISE_MACROBLOCKS_BLOCK_DIM_X * blockIdx.x );
	i_tid = threadIdx.x;

	if( i_tid < 64 )
	{
		rgui16_noise_offset[ 0 ][ i_tid ] = ps_denoise->rgi_intra_signal_offset[ i_tid ];
		rgui16_noise_offset[ 1 ][ i_tid ] = ps_denoise->rgi_inter_signal_offset[ i_tid ];
	}
	SYNC;

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	if( ps_mb )
	{
		if( ps_mb->i_macroblock_type == H261_MB_TYPE_INTRA )
		{
			rgi_noise_accum[ 0 ][ i_tid ] = 1;
			rgi_noise_accum[ 1 ][ i_tid ] = 0;

			i_block_idx = 0;
		}
		else
		{
			rgi_noise_accum[ 0 ][ i_tid ] = 0;
			rgi_noise_accum[ 1 ][ i_tid ] = 1;

			i_block_idx = 1;
		}
	}
	else
	{
		rgi_noise_accum[ 0 ][ i_tid ] = 0;
		rgi_noise_accum[ 1 ][ i_tid ] = 0;
		i_block_idx = 0;
	}

	SYNC;
	for( i_stride = DENOISE_MACROBLOCKS_BLOCK_DIM_X / 2; i_stride > 0; i_stride >>= 1 )
	{
		if( i_tid < i_stride )
		{
			rgi_noise_accum[ 0 ][ i_tid ] += rgi_noise_accum[ 0 ][ i_tid + i_stride ];
			rgi_noise_accum[ 1 ][ i_tid ] += rgi_noise_accum[ 1 ][ i_tid + i_stride ];
		}
		SYNC;
	}
	if( i_tid == 0 )
	{
		i_intra = rgi_noise_accum[ 0 ][ i_tid ];
		i_inter = rgi_noise_accum[ 1 ][ i_tid ];
		atomicAdd( &ps_denoise->i_intra_count, i_intra );
		atomicAdd( &ps_denoise->i_inter_count, i_inter );
	}


	i_idx = 0;
	for( i_y = 0; i_y < 8; i_y++ )
	{
		for( i_x = 0; i_x < 8; i_x++ )
		{
			Int32 i_sign, i_noise_intra, i_noise_inter, i_signal_offset;

			i_signal_offset = rgi_noise_accum[ i_block_idx ][ 0 ];
			i_signal_offset = rgui16_noise_offset[ i_block_idx ][ i_idx ];

			if( ps_mb )
			{
				i_level = pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ];

				i_sign = i_level >> ( sizeof( Int32 ) * 8 - 1 );
				i_level = ( i_level ^ i_sign ) - i_sign;

				rgi_noise_accum[ i_block_idx ][ i_tid ] = i_level;
				rgi_noise_accum[ !i_block_idx ][ i_tid ] = 0;

				i_level -= i_signal_offset;
				i_level = max( i_level, 0 );

				i_level = ( i_level ^ i_sign ) - i_sign;

				pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ] = i_level;
			}
			else
			{
				rgi_noise_accum[ 0 ][ i_tid ] = 0;
				rgi_noise_accum[ 1 ][ i_tid ] = 0;
			}


			SYNC;

			for( i_stride = DENOISE_MACROBLOCKS_BLOCK_DIM_X / 2; i_stride > 0; i_stride >>= 1 )
			{
				if( i_tid < i_stride )
				{
					rgi_noise_accum[ 0 ][ i_tid ] += rgi_noise_accum[ 0 ][ i_tid + i_stride ];
					rgi_noise_accum[ 1 ][ i_tid ] += rgi_noise_accum[ 1 ][ i_tid + i_stride ];
				}
				SYNC;
			}

			if( i_tid == 0 )
			{
				i_noise_intra = rgi_noise_accum[ 0 ][ i_tid ];
				i_noise_inter = rgi_noise_accum[ 1 ][ i_tid ];
				atomicAdd( &ps_denoise->rgi_intra_noise[ i_idx ], i_noise_intra );
				atomicAdd( &ps_denoise->rgi_inter_noise[ i_idx ], i_noise_inter );
			}
			i_idx++;
		}
		pi_coeffs += COEFF_BLOCK_STRIDE_Y;
	}
}


/* quantisation */

#define QUANT_MACROBLOCKS_BLOCK_DIM_X ( 128 )
#define QUANT_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

Void __global__ macroblocks_quant_forward( )
{
	Int32 *pi_coeffs, *pi_coeffs2;
	Int32 i_coeff_id, i_coeff_idx;
	Int32 i_x, i_dc, i_quant, i_level, i_coeff_accum, i_start;
	float f_inv_quant, f_level;

	const Int32 rgi_cbp_table[ 6 ] = {
		H261_CODED_BLOCK_PATTERN_1,
		H261_CODED_BLOCK_PATTERN_2,
		H261_CODED_BLOCK_PATTERN_3,
		H261_CODED_BLOCK_PATTERN_4,
		H261_CODED_BLOCK_PATTERN_5,
		H261_CODED_BLOCK_PATTERN_6
	};

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( QUANT_MACROBLOCKS_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	if( ps_mb )
	{
		i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
		pi_coeffs2 = &ps_mb->rgi_tcoeff[ i_coeff_idx ][ 0 ];

		i_quant = ps_mb->i_macroblock_quant;
		if( i_coeff_idx == 0 )
		{
			ps_mb->i_coded_block_pattern = 0;
		}
	}
	SYNC;

	if( ps_mb != 0 )
	{
		f_inv_quant = 1.0 / ( float ) ( i_quant * 2 );

		i_coeff_accum = 0;
		
		if( ps_mb->i_macroblock_type == H261_MB_TYPE_INTRA )
		{
			i_dc = pi_coeffs[ 0 ] / 8;
			i_dc = max( i_dc, 1 );
			i_dc = min( i_dc, 254 );
			if( i_dc == 128 )
			{
				i_dc = 255;
			}
			pi_coeffs[ 0 ] = i_dc;
			pi_coeffs2[ 0 ] = i_dc;
			i_coeff_accum = 100;
			i_start = 1;
		}
		else
		{
			i_start = 0;
		}

		for( i_x = i_start; i_x < 64; i_x++ )
		{
			f_level = ( float ) pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ] * f_inv_quant;
			i_level = min( 127, max( -127, ( Int32 ) f_level ) );
			pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ] = i_level;
			pi_coeffs2[ i_x ] = i_level;
			i_level = abs( i_level );
			i_coeff_accum += i_level > 1 ? 100 : i_level;
		}

		if( i_coeff_accum > 2 )
		{
			atomicOr( &ps_mb->i_coded_block_pattern, rgi_cbp_table[ i_coeff_idx ] );
			atomicOr( &ps_mb->i_macroblock_type_flags, H261_MB_COEFF );
		}
	}
}


/* de- quantisation */
#define DEQUANT_MACROBLOCKS_BLOCK_DIM_X 128
#define DEQUANT_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

Void __global__ macroblocks_quant_backward( )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx;
	Int32 i_x, i_dc, i_quant, i_level, i_quant_mul, i_quant_add, i_start;

	const Int32 rgi_cbp_table[ 6 ] = {
		H261_CODED_BLOCK_PATTERN_1,
		H261_CODED_BLOCK_PATTERN_2,
		H261_CODED_BLOCK_PATTERN_3,
		H261_CODED_BLOCK_PATTERN_4,
		H261_CODED_BLOCK_PATTERN_5,
		H261_CODED_BLOCK_PATTERN_6
	};

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( DEQUANT_MACROBLOCKS_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );

	i_quant = ps_mb->i_macroblock_quant;
	i_quant_add = ( i_quant - 1 ) | 1;
	i_quant_mul = i_quant * 2;

	{
		volatile const Int32 *pi_cbp_table;
		Int32 i_table;
		pi_cbp_table = &rgi_cbp_table[ 0 ];
		pi_cbp_table += i_coeff_idx;
		i_table = *pi_cbp_table;
		if( ! ( i_table & ps_mb->i_coded_block_pattern ) )
		{
			i_quant = 0;
			i_quant_add = 0;
			i_quant_mul = 0;
		}
	}

	if( ps_mb->i_macroblock_type == H261_MB_TYPE_INTRA )
	{
		i_dc = pi_coeffs[ 0 ];
		if( i_dc == 255 )
		{
			i_dc = 1024;
		}
		else
		{
			i_dc *= 8;
		}
		pi_coeffs[ 0 ] = i_dc;
		i_start = 1;
	}
	else
	{
		i_start = 0;
	}

	for( i_x = i_start; i_x < 64; i_x++ )
	{
		i_level = pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ];
		i_level = i_level * i_quant_mul;
		if( i_level > 0 )
		{
			i_level += i_quant_add;
		}
		else if( i_level < 0 )
		{
			i_level -= i_quant_add;
		}
		i_level = min( 2047, max( -2048, i_level ) );
		pi_coeffs[ i_x * COEFF_BLOCK_STRIDE_X ] = i_level;
	}
}


/* inverse dct */

#define IDCT_MACROBLOCKS_BLOCK_DIM_X ( 288 )
#define IDCT_MACROBLOCKS_BLOCK_DIM_Y ( 1 )

float __constant__ g_rgf_inverse_prescale[ 8 ] = { 1.0, 1.387040, 1.306563, 1.175876, 1.0, 0.785695, 0.541196, 0.275899 };

#define CLAMP_255( x ) ( min( 255, max( -256, ( x ) ) ) )

Void __device__ macroblocks_idct8x8( Int32 *pi_coeffs )
{
	Int32 i_idx;
	float tmp0, tmp1, tmp2, tmp3;
	float tmp10, tmp11, tmp12, tmp13;
	float z10, z11, z12, z13, z5;
	float rgf_temp[ 64 ];
	float f_column_scale;

	for( i_idx = 0; i_idx < 8; i_idx++ )
	{
		f_column_scale = g_rgf_inverse_prescale[ i_idx ] * 0.125;
		rgf_temp[ i_idx * 8 + 0 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 0 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 0 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 1 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 1 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 1 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 2 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 2 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 2 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 3 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 3 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 3 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 4 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 4 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 4 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 5 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 5 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 5 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 6 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 6 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 6 ] * f_column_scale;
		rgf_temp[ i_idx * 8 + 7 ] = ( ( float ) pi_coeffs[ ( i_idx * COEFF_BLOCK_STRIDE_Y ) + 7 * COEFF_BLOCK_STRIDE_X ] ) * g_rgf_inverse_prescale[ 7 ] * f_column_scale;
	}

	for( i_idx = 0; i_idx < 8; i_idx++ )
	{
		tmp12 = rgf_temp[ 16 + i_idx ] - rgf_temp[ 48 + i_idx ];
		tmp13 = rgf_temp[ 16 + i_idx ] + rgf_temp[ 48 + i_idx ];
		tmp12 *= 1.414214;
		tmp12 -= tmp13;

		tmp10 = rgf_temp[ 0 + i_idx ] + rgf_temp[ 32 + i_idx ];
		tmp11 = rgf_temp[ 0 + i_idx ] - rgf_temp[ 32 + i_idx ];

		tmp0 = tmp10 + tmp13;
		tmp1 = tmp11 + tmp12;
		tmp2 = tmp11 - tmp12;
		tmp3 = tmp10 - tmp13;

		z11 = rgf_temp[ 8 + i_idx ] + rgf_temp[ 56 + i_idx ];
		z12 = rgf_temp[ 8 + i_idx ] - rgf_temp[ 56 + i_idx ];
		z13 = rgf_temp[ 40 + i_idx ] + rgf_temp[ 24 + i_idx ];
		z10 = rgf_temp[ 40 + i_idx ] - rgf_temp[ 24 + i_idx ];

		tmp13 =  z11 + z13;
		tmp11 = ( z11 - z13 ) * 1.414214;

		z5 = ( z12 + z10 ) * 1.84776;
		tmp10 =  ( z12 * 1.082392 ) - z5;
		tmp12 =  ( z10 * -2.613126 ) + z5;

		tmp12 -= tmp13;
		tmp11 -= tmp12;
		tmp10 += tmp11;

		rgf_temp[ 0 + i_idx ] = tmp0 + tmp13;
		rgf_temp[ 8 + i_idx ] = tmp1 + tmp12;
		rgf_temp[ 16 + i_idx ] = tmp2 + tmp11;
		rgf_temp[ 24 + i_idx ] = tmp3 - tmp10;
		rgf_temp[ 32 + i_idx ] = tmp3 + tmp10;
		rgf_temp[ 40 + i_idx ] = tmp2 - tmp11;
		rgf_temp[ 48 + i_idx ] = tmp1 - tmp12;
		rgf_temp[ 56 + i_idx ] = tmp0 - tmp13;
	}

	for( i_idx = 0; i_idx < 64; i_idx += 8 )
	{
		tmp13 = rgf_temp[ 2 + i_idx ] + rgf_temp[ 6 + i_idx ];
		tmp12 = rgf_temp[ 2 + i_idx ] - rgf_temp[ 6 + i_idx ];
		tmp12 *= 1.414214;
		tmp12 -= tmp13;

		tmp10 = rgf_temp[ 0 + i_idx ] + rgf_temp[ 4 + i_idx ];
		tmp11 = rgf_temp[ 0 + i_idx ] - rgf_temp[ 4 + i_idx ];

		tmp0 = tmp10 + tmp13;
		tmp1 = tmp11 + tmp12;
		tmp2 = tmp11 - tmp12;
		tmp3 = tmp10 - tmp13;

		z11 = rgf_temp[ 1 + i_idx ] + rgf_temp[ 7 + i_idx ];
		z12 = rgf_temp[ 1 + i_idx ] - rgf_temp[ 7 + i_idx ];
		z13 = rgf_temp[ 5 + i_idx ] + rgf_temp[ 3 + i_idx ];
		z10 = rgf_temp[ 5 + i_idx ] - rgf_temp[ 3 + i_idx ];

		tmp13 =  z11 + z13;
		tmp11 = ( z11 - z13 ) * 1.414214;

		z5 = ( z12 + z10 ) * 1.84776;
		tmp10 =  ( z12 * 1.082392 ) - z5;
		tmp12 =  ( z10 * -2.613126 ) + z5;

		tmp12 -= tmp13;
		tmp11 -= tmp12;
		tmp10 += tmp11;

		pi_coeffs[ 0 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp0 + tmp13 + 0.49999 ) ) );
		pi_coeffs[ 1 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp1 + tmp12 + 0.49999 ) ) );
		pi_coeffs[ 2 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp2 + tmp11 + 0.49999 ) ) );
		pi_coeffs[ 3 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp3 - tmp10 + 0.49999 ) ) );
		pi_coeffs[ 4 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp3 + tmp10 + 0.49999 ) ) );
		pi_coeffs[ 5 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp2 - tmp11 + 0.49999 ) ) );
		pi_coeffs[ 6 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp1 - tmp12 + 0.49999 ) ) );
		pi_coeffs[ 7 * COEFF_BLOCK_STRIDE_X ] = CLAMP_255( ( Int32 ) floor( ( tmp0 - tmp13 + 0.49999 ) ) );
		pi_coeffs += COEFF_BLOCK_STRIDE_Y;
	}
}


Void __global__ macroblocks_dct_inverse( )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	const Int32 rgi_cbp_table[ ] = {
		H261_CODED_BLOCK_PATTERN_1,
		H261_CODED_BLOCK_PATTERN_2,
		H261_CODED_BLOCK_PATTERN_3,
		H261_CODED_BLOCK_PATTERN_4,
		H261_CODED_BLOCK_PATTERN_5,
		H261_CODED_BLOCK_PATTERN_6
	};

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( IDCT_MACROBLOCKS_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );


	{
		volatile const Int32 *pi_cbp_table;
		Int32 i_table;
		pi_cbp_table = &rgi_cbp_table[ 0 ];
		pi_cbp_table += i_coeff_idx;
		i_table = *pi_cbp_table;
		if( ! ( i_table & ps_mb->i_coded_block_pattern ) )
		{
			return;
		}
	}

	macroblocks_idct8x8( pi_coeffs );
}


/* reconstruction */

#define RECON_MACROBLOCKS_LUMA_BLOCK_DIM_X 128
#define RECON_MACROBLOCKS_LUMA_BLOCK_DIM_Y 1

#define CLAMP_255U( x ) ( min( 255, max( 0, ( x ) ) ) )
Void __device__ macroblocks_reconstruct_intra( UInt8 *pui8_destination, Int32 i_destination_stride, Int32 *pi_coeffs )
{
	Int32 i_x, i_y;

	for( i_y = 0; i_y < COEFF_SIZE; i_y++ )
	{
		for( i_x = 0; i_x < COEFF_SIZE; i_x++ )
		{
			Int32 i_pel;
			
			i_pel = pi_coeffs[ ( i_y * COEFF_BLOCK_STRIDE_Y ) + ( i_x * COEFF_BLOCK_STRIDE_X ) ];
			i_pel = CLAMP_255U( i_pel );
			pui8_destination[ i_y * i_destination_stride + i_x ] = i_pel;
		}
	}
}


Void __device__ macroblocks_reconstruct_inter( cudaTextureObject_t ps_reference_texture, UInt8 *pui8_destination, Int32 i_destination_stride, Int32 *pi_coeffs, Int32 i_mb_flags, Int32 i_mv_x, Int32 i_mv_y )
{
	Int32 i_x, i_y;

	if( i_mb_flags & H261_MB_FILTER )
	{
		Int32 i_pel1, i_pel2;
		Int32 rgi_temp[ 64 ];

		for( i_x = 0; i_x < 8; i_x++ )
		{
			Int32 i_a1, i_a2, i_a3;

			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 0 );
			rgi_temp[ 0 * 8 + i_x ] = i_a1 * 4;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 1 );
			i_a3 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 2 );
			rgi_temp[ 1 * 8 + i_x ] = i_a1 + i_a2 * 2 + i_a3;
			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 3 );
			rgi_temp[ 2 * 8 + i_x ] = i_a2 + i_a3 * 2 + i_a1;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 4 );
			rgi_temp[ 3 * 8 + i_x ] = i_a3 + i_a1 * 2 + i_a2;
			i_a3 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 5 );
			rgi_temp[ 4 * 8 + i_x ] = i_a1 + i_a2 * 2 + i_a3;
			i_a1 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 6 );
			rgi_temp[ 5 * 8 + i_x ] = i_a2 + i_a3 * 2 + i_a1;
			i_a2 = tex2D<unsigned char>( ps_reference_texture, (float) i_mv_x + i_x, (float) i_mv_y + 7 );
			rgi_temp[ 6 * 8 + i_x ] = i_a3 + i_a1 * 2 + i_a2;
			rgi_temp[ 7 * 8 + i_x ] = i_a2 * 4;
		}
		for( i_y = 0; i_y < 8; i_y++ )
		{
			Int32 i_a1, i_a2, i_a3;

			i_a1 = rgi_temp[ 8 * i_y + 0 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 0 * COEFF_BLOCK_STRIDE_X];
			i_pel2 = ( i_a1 + 2 ) >> 2;
			pui8_destination[ i_y * i_destination_stride + 0 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a2 = rgi_temp[ 8 * i_y + 1 ];
			i_a3 = rgi_temp[ 8 * i_y + 2 ];

			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 1 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a1 + i_a2 * 2 + i_a3 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 1 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a1 = rgi_temp[ 8 * i_y + 3 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 2 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a2 + i_a3 * 2 + i_a1 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 2 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a2 = rgi_temp[ 8 * i_y + 4 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 3 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a3 + i_a1 * 2 + i_a2 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 3 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a3 = rgi_temp[ 8 * i_y + 5 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 4 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a1 + i_a2 * 2 + i_a3 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 4 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a1 = rgi_temp[ 8 * i_y + 6 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 5 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a2 + i_a3 * 2 + i_a1 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 5 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_a2 = rgi_temp[ 8 * i_y + 7 ];
			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 6 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a3 + i_a1 * 2 + i_a2 + 8 ) >> 4;
			pui8_destination[ i_y * i_destination_stride + 6 ] = CLAMP_255U( i_pel1 + i_pel2 );

			i_pel1 = pi_coeffs[ COEFF_BLOCK_STRIDE_Y * i_y + 7 * COEFF_BLOCK_STRIDE_X ];
			i_pel2 = ( i_a2 + 2 ) >> 2;
			pui8_destination[ i_y * i_destination_stride + 7 ] = CLAMP_255U( i_pel1 + i_pel2 );
		}
	}
	else
	{
		for( i_y = 0; i_y < COEFF_SIZE; i_y++ )
		{
			for( i_x = 0; i_x < COEFF_SIZE; i_x++ )
			{
				Int32 i_pel1, i_pel2;

				i_pel1 = pi_coeffs[ i_y * COEFF_BLOCK_STRIDE_Y + i_x * COEFF_BLOCK_STRIDE_X ];
				i_pel2 = tex2D<unsigned char>( ps_reference_texture, (float) i_x + i_mv_x, (float) i_y + i_mv_y );
				pui8_destination[ i_y * i_destination_stride + i_x ] = CLAMP_255U( i_pel1 + i_pel2 );
			}
		}
	}
}


Void __global__ macroblocks_reconstruct_luma( cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_x, i_coeff_y, i_coeff_idx;
	Int32 i_mb_flags;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	UInt8 *pui8_destination;
	Int32 i_destination_stride;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( RECON_MACROBLOCKS_LUMA_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx > 3 )
	{
		return;
	}

	i_mb_flags = ps_mb->i_macroblock_type_flags;
	
	i_mv_x = ps_mb->rgi_mv[ 0 ];
	i_mv_y = ps_mb->rgi_mv[ 1 ];

	i_coeff_x = i_coeff_idx & 1;
	i_coeff_y = ( i_coeff_idx & 2 ) >> 1;

	i_pel_x = ps_mb->i_mb_x + ( i_coeff_x * 8 );
	i_pel_y = ps_mb->i_mb_y + ( i_coeff_y * 8 );

	i_destination_stride = ps_mbs->i_mb_width * 16;
	pui8_destination = ps_mbs->pui8_reconstructed_Y_device;
	pui8_destination += i_pel_x + ( i_pel_y * i_destination_stride );

	i_mv_x += i_pel_x;
	i_mv_y += i_pel_y;

	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_reconstruct_intra( pui8_destination, i_destination_stride, pi_coeffs );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_reconstruct_inter( ps_reference_texture, pui8_destination, i_destination_stride, pi_coeffs, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}


#define RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_X 128
#define RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_Y 1

Void __global__ macroblocks_reconstruct_chromab( cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx, i_mb_flags;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	UInt8 *pui8_destination;
	Int32 i_destination_stride;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx != 4 )
	{
		return;
	}

	i_mb_flags = ps_mb->i_macroblock_type_flags;

	i_mv_x = ps_mb->rgi_mv[ 0 ] / 2;
	i_mv_y = ps_mb->rgi_mv[ 1 ] / 2;

	i_pel_x = ps_mb->i_mb_x / 2;
	i_pel_y = ps_mb->i_mb_y / 2;

	i_destination_stride = ps_mbs->i_mb_width * 8;
	pui8_destination = ps_mbs->pui8_reconstructed_Cb_device;
	pui8_destination += i_pel_x + ( i_pel_y * i_destination_stride );

	i_mv_x += i_pel_x;
	i_mv_y += i_pel_y;

	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_reconstruct_intra( pui8_destination, i_destination_stride, pi_coeffs );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_reconstruct_inter( ps_reference_texture, pui8_destination, i_destination_stride, pi_coeffs, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}


Void __global__ macroblocks_reconstruct_chromar( cudaTextureObject_t ps_reference_texture )
{
	Int32 *pi_coeffs;
	Int32 i_coeff_id, i_coeff_idx, i_mb_flags;
	Int32 i_pel_x, i_pel_y, i_mv_x, i_mv_y;

	UInt8 *pui8_destination;
	Int32 i_destination_stride;

	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;

	ps_mbs = &g_macroblocks_constant_gpu_device;

	i_coeff_id = threadIdx.x + ( RECON_MACROBLOCKS_CHROMA_BLOCK_DIM_X * blockIdx.x );

	macroblocks_get_macroblock_ptr( ps_mbs, i_coeff_id, &ps_mb );
	if( ps_mb == 0 )
	{
		return;
	}

	macroblocks_get_coeffs_ptr( ps_mbs, i_coeff_id, &pi_coeffs );

	i_coeff_idx = macroblocks_get_coeff_idx( i_coeff_id );
	if( i_coeff_idx != 5 )
	{
		return;
	}

	i_mb_flags = ps_mb->i_macroblock_type_flags;

	i_mv_x = ps_mb->rgi_mv[ 0 ] / 2;
	i_mv_y = ps_mb->rgi_mv[ 1 ] / 2;

	i_pel_x = ps_mb->i_mb_x / 2;
	i_pel_y = ps_mb->i_mb_y / 2;

	i_destination_stride = ps_mbs->i_mb_width * 8;
	pui8_destination = ps_mbs->pui8_reconstructed_Cr_device;
	pui8_destination += i_pel_x + ( i_pel_y * i_destination_stride );

	i_mv_x += i_pel_x;
	i_mv_y += i_pel_y;

	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		macroblocks_reconstruct_intra( pui8_destination, i_destination_stride, pi_coeffs );
	break;

	case H261_MB_TYPE_INTER:
		macroblocks_reconstruct_inter( ps_reference_texture, pui8_destination, i_destination_stride, pi_coeffs, i_mb_flags, i_mv_x, i_mv_y );
	break;
	}
}

