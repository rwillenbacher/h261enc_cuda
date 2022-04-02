
#include "h261_decl.h"

void usage( Int8 *rgpui8_argv[] )
{
	printf("%s: <cif file> <bitstream file> <recon file>", rgpui8_argv[0] );
}


Void h261_init_frame( h261_frame_t *ps_frame, Int32 i_width, Int32 i_height )
{
	Int32 i_frame_size_y, i_frame_size_c;

	ps_frame->i_width = i_width;
	ps_frame->i_height = i_height;
	ps_frame->i_stride_y = i_width;
	ps_frame->i_stride_c = i_width / 2;

	i_frame_size_y = ps_frame->i_stride_y * ps_frame->i_height;
	i_frame_size_c = ps_frame->i_stride_c * ( ps_frame->i_height / 2 );

	ps_frame->pui8_Y = malloc( i_frame_size_y );
	ps_frame->pui8_Cb = malloc( i_frame_size_c );
	ps_frame->pui8_Cr = malloc( i_frame_size_c );

	memset( ps_frame->pui8_Y, 0, i_frame_size_y );
	memset( ps_frame->pui8_Cb, 0, i_frame_size_c );
	memset( ps_frame->pui8_Cr, 0, i_frame_size_c );
}

Void h261_deinit_frame( h261_frame_t *ps_frame )
{
	free( ps_frame->pui8_Y );
	free( ps_frame->pui8_Cb );
	free( ps_frame->pui8_Cr );
}

Void get_frame_psnr( h261_frame_t *ps_original, h261_frame_t *ps_recon, double *pd_psnr )
{
	Int32 i_x, i_y, i_num_pel_x, i_num_pel_y, i_pel_stride, i_idx;
	UInt64 ui64_ssd;
	UInt8 *pui8_ref, *pui8_recon;

	for( i_idx = 0; i_idx < 3; i_idx++ )
	{
		if( i_idx == 0 )
		{
			pui8_ref = ps_original->pui8_Y;
			pui8_recon = ps_recon->pui8_Y;
			i_num_pel_x = ps_original->i_width;
			i_num_pel_y = ps_original->i_height;
			i_pel_stride = ps_original->i_stride_y;
		}
		else
		{
			if( i_idx == 1 )
			{
				pui8_ref = ps_original->pui8_Cb;
				pui8_recon = ps_recon->pui8_Cb;
			}
			else
			{
				pui8_ref = ps_original->pui8_Cr;
				pui8_recon = ps_recon->pui8_Cr;
			}
			i_num_pel_x = ps_original->i_width / 2;
			i_num_pel_y = ps_original->i_height / 2;
			i_pel_stride = ps_original->i_stride_c;
		}
		ui64_ssd = 0;
		for( i_y = 0; i_y < i_num_pel_y; i_y++ )
		{
			for( i_x = 0; i_x < i_num_pel_x; i_x++ )
			{
				UInt32 ui_delta;
				ui_delta = pui8_ref[ i_x ] - pui8_recon[ i_x ];
				ui64_ssd += ui_delta * ui_delta;
			}
			pui8_ref += i_pel_stride;
			pui8_recon += i_pel_stride;
		}
	    if( 0 == ui64_ssd )
	    {
			pd_psnr[ i_idx ] = 100.0;
		}
		else 
		{
	        pd_psnr[ i_idx ] = 10 * log10( 255.0 * 255.0 * ( i_num_pel_x * i_num_pel_y ) / ( double )( ui64_ssd ) );
	    }
	}
}



int main( Int32 i_argc, Int8 *rgpui8_argv[] )
{
	UInt8 *pui8_bitstream;
	Int32 i_width, i_height, i_frame_size, i_frame_counter, i_bitstream_length;
	Int32 i_sequence_size;
	double d_start_time, d_end_time, d_global_start_time, d_global_end_time, rgd_psnr[ 3 ], rgd_psnr_mean[ 3 ];
	FILE *f, *fout, *frecout;

	h261_frame_t *ps_source_frame;
	h261_frame_t *ps_reference_frame;
	h261_context_t *ps_context;
	h261_bitstream_t *ps_bitstream;

	me_gpu_t *ps_me_gpu;
	h261_mode_decision_t *ps_md;
	h261_macroblocks_t *ps_mbs;

	if( i_argc < 4 )
	{
		usage( rgpui8_argv );
		return 1;
	}

	f = fopen( rgpui8_argv[ 1 ], "rb" );

	if( !f )
	{
		usage( rgpui8_argv );
		return 1;
	}

	fout = fopen( rgpui8_argv[ 2 ], "wb");
	if( !fout )
	{
		usage( rgpui8_argv );
		return 2;
	}

	frecout = fopen( rgpui8_argv[ 3 ], "wb" );
	if( !frecout )
	{
		usage( rgpui8_argv );
		return 3;
	}

	ps_context = malloc( sizeof( h261_context_t ) );
	memset( ps_context, 0, sizeof( h261_context_t ) );

	ps_context->i_source_format = H261_SOURCE_FORMAT_CIF;
	i_width = 352;
	i_height = 288;
	i_frame_size = i_width * i_height;

	/* frame init */
	ps_source_frame = &ps_context->s_current_frame;
	ps_reference_frame = &ps_context->s_reference_frame;

	h261_init_frame( ps_source_frame, i_width, i_height );
	h261_init_frame( ps_reference_frame, i_width, i_height );

	/* bitstream init */
	ps_bitstream = &ps_context->s_bitstream;
	h261_bitstream_init( ps_bitstream, 0x100000 );

	/* encoder core init */
	h261_gpu_init();
	h261_gpu_device_init_textures( i_width, i_height );
	h261_init_macroblocks( &ps_mbs, i_width, i_height, 5 );
	h261_gpu_device_init_me( &ps_me_gpu, i_width, i_height );
	h261_init_mode_decision( &ps_md, ps_me_gpu->i_mb_width, ps_me_gpu->i_mb_height );
	h261_init_ratectrl( ps_context, 10, 512000, 512000, H261_RATECTRL_MODE_CBR );

	i_frame_counter = 0;

	rgd_psnr_mean[ 0 ] = rgd_psnr_mean[ 1 ] = rgd_psnr_mean[ 2 ] = 0.0;
	i_sequence_size = 0;
	d_global_start_time = h261_get_time();

	while( 1 )
	{
		Int32 i_ret;
		h261_picture_parameters_t s_picture_parameters;

		fread( ps_source_frame->pui8_Y, i_frame_size, 1, f );
		fread( ps_source_frame->pui8_Cb, i_frame_size / 4, 1, f );
		i_ret = fread( ps_source_frame->pui8_Cr, i_frame_size / 4, 1, f );
		if( i_ret <= 0 )
		{
			break;
		}


		d_start_time = h261_get_time();

		/* textures */
		h261_gpu_device_set_current( ps_source_frame->pui8_Y, i_width, i_height );
		h261_gpu_device_set_current_chroma_cb( ps_source_frame->pui8_Cb, i_width / 2, i_height / 2 );
		h261_gpu_device_set_current_chroma_cr( ps_source_frame->pui8_Cr, i_width / 2, i_height / 2 );

		h261_gpu_device_set_reference( ps_reference_frame->pui8_Y, i_width, i_height );
		h261_gpu_device_set_reference_chroma_cb( ps_reference_frame->pui8_Cb, i_width / 2, i_height / 2 );
		h261_gpu_device_set_reference_chroma_cr( ps_reference_frame->pui8_Cr, i_width / 2, i_height / 2 );


		/* motion estimation */
		h261_gpu_device_me( ps_me_gpu );

		/* init picture */
		s_picture_parameters.i_frame_num = i_frame_counter;
		s_picture_parameters.i_source_format = ps_context->i_source_format;

		if( i_frame_counter == 0 )
		{
			s_picture_parameters.i_frame_type = H261_FRAME_TYPE_INTRA;
		}
		else
		{
			s_picture_parameters.i_frame_type = H261_FRAME_TYPE_INTER;
		}

		/* mode decision */
		h261_mode_decision( ps_md, ps_me_gpu, &s_picture_parameters );
		ps_context->ps_mode_decision = ps_md;

		/* ratectrl */
		h261_ratectrl_collect_picture_stats( ps_context, &s_picture_parameters );
		h261_ratectrl_init_picture( ps_context, &s_picture_parameters );

		/* encode macroblocks */
		h261_init_picture( ps_context, &s_picture_parameters );

		ps_context->ps_macroblocks = ps_mbs;

		h261_gpu_device_encode_macroblocks( ps_context );

		/* bitstream writer */
		h261_bitstream_reset( ps_bitstream );
		h261_write_picture_header( ps_context );
		h261_write_gobs( ps_context );

		/* recon */
		h261_gpu_device_decode_macroblocks( ps_context );

		/* bitstream out */
		h261_bitstream_get( ps_bitstream, &pui8_bitstream, &i_bitstream_length );
		fwrite( pui8_bitstream, i_bitstream_length, 1, fout );

		/* update functions */
		h261_ratectrl_update_picture( ps_context, i_bitstream_length );

		d_end_time = h261_get_time();

		memcpy( ps_reference_frame->pui8_Y, ps_context->ps_macroblocks->pui8_reconstructed_Y_host, i_frame_size );
		memcpy( ps_reference_frame->pui8_Cb, ps_context->ps_macroblocks->pui8_reconstructed_Cb_host, i_frame_size / 4 );
		memcpy( ps_reference_frame->pui8_Cr, ps_context->ps_macroblocks->pui8_reconstructed_Cr_host, i_frame_size / 4 );

		get_frame_psnr( ps_source_frame, ps_reference_frame, &rgd_psnr[ 0 ] );

		fwrite( ps_reference_frame->pui8_Y, i_frame_size, 1, frecout );
		fwrite( ps_reference_frame->pui8_Cb, i_frame_size / 4, 1, frecout );
		fwrite( ps_reference_frame->pui8_Cr, i_frame_size / 4, 1, frecout );

		printf("frame %04d, quant=%d, cost=%d, size=%d, ypsnr=%f, cbpsnr=%f, crpsnr=%f, time=%.3fms\n",
			i_frame_counter,
			s_picture_parameters.i_quantiser,
			ps_context->s_ratectrl.i_current_picture_cost,
			i_bitstream_length,
			rgd_psnr[ 0 ], rgd_psnr[ 1 ], rgd_psnr[ 2 ],
			d_end_time - d_start_time );

		rgd_psnr_mean[ 0 ] += rgd_psnr[ 0 ];
		rgd_psnr_mean[ 1 ] += rgd_psnr[ 1 ];
		rgd_psnr_mean[ 2 ] += rgd_psnr[ 2 ];
		i_sequence_size += i_bitstream_length;

		i_frame_counter++;
	}
	d_global_end_time = h261_get_time();

	printf("PSNR Mean: %f, %f, %f\n",
		rgd_psnr_mean[ 0 ] / i_frame_counter,
		rgd_psnr_mean[ 1 ] / i_frame_counter,
		rgd_psnr_mean[ 2 ] / i_frame_counter );
	printf("%f kBit/s, ", ( ( double ) i_sequence_size * 8 ) / ( ( double )i_frame_counter / 0.02997 ) );
	printf("%f fps\n", ( ( double ) i_frame_counter ) / ( ( d_global_end_time - d_global_start_time ) / 1000.0 ) );


	h261_gpu_device_deinit_textures( );
	h261_gpu_device_deinit_me( &ps_me_gpu );
	h261_deinit_mode_decision( ps_md );
	h261_deinit_macroblocks( ps_mbs );

	h261_bitstream_deinit( ps_bitstream );
	h261_deinit_frame( ps_source_frame );
	h261_deinit_frame( ps_reference_frame );

	fclose( f );
	fclose( fout );
	fclose( frecout );

	return 0;
}



/*
	{
		Int32 i_idx, i_idx2;
		const Int32 rgi_pels[ 64 ] = {
			1,2,3,4,5,6,7,8,
			9,10,11,12,13,14,15,16,
			17,18,19,20,21,22,23,24,
			25,26,27,28,29,30,31,32,
			33,34,35,36,37,38,39,40,
			41,42,43,44,45,46,47,48,
			49,50,51,52,53,54,55,56,
			57,58,59,60,61,62,63,64
		};

		Int32 rgi_coeffs[ 64 ], rgi_recon_pels[ 64 ];

		dct8x8_fw( &rgi_pels[ 0 ], 8, &rgi_coeffs[ 0 ], 8 );
		dct8x8_bw( &rgi_coeffs[ 0 ], 8, &rgi_recon_pels[ 0 ], 8 );

		printf("pels:\n");
		for( i_idx = 0; i_idx < 8; i_idx++ )
		{
			for( i_idx2 = 0; i_idx2 < 8; i_idx2++ )
			{
				printf( "%d, ", rgi_pels[ i_idx * 8 + i_idx2 ] );
			}
			printf("\n");
		}
		printf("coeffs:\n");
		for( i_idx = 0; i_idx < 8; i_idx++ )
		{
			for( i_idx2 = 0; i_idx2 < 8; i_idx2++ )
			{
				printf( "%d, ", rgi_coeffs[ i_idx * 8 + i_idx2 ] );
			}
			printf("\n");
		}
		printf("recon_pels:\n");
		for( i_idx = 0; i_idx < 8; i_idx++ )
		{
			for( i_idx2 = 0; i_idx2 < 8; i_idx2++ )
			{
				printf( "%d, ", rgi_recon_pels[ i_idx * 8 + i_idx2 ] );
			}
			printf("\n");
		}
	}
*/
