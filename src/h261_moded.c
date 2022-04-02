
#include "h261_decl.h"


Void h261_init_mode_decision( h261_mode_decision_t **pps_md, Int32 i_mb_width, Int32 i_mb_height )
{
	Int32 i_num_mb;
	h261_mode_decision_t *ps_md;

	i_num_mb = i_mb_width * i_mb_height;

	ps_md = malloc( sizeof( h261_mode_decision_t ) );
	memset( ps_md, 0, sizeof( h261_mode_decision_t ) );

	ps_md->prgs_mb = malloc( sizeof( h261_mb_mode_decision_t ) * i_num_mb );
	memset( ps_md->prgs_mb, 0, sizeof( h261_mb_mode_decision_t ) * i_num_mb );
	ps_md->i_mb_width = i_mb_width;
	ps_md->i_mb_height = i_mb_height;

	*pps_md = ps_md;
}


Void h261_deinit_mode_decision( h261_mode_decision_t *ps_md )
{
	free( ps_md->prgs_mb );
	free( ps_md );
}


Void h261_mode_decision( h261_mode_decision_t *ps_md, me_gpu_t *ps_me_gpu, h261_picture_parameters_t *ps_picture )
{
	Int32 i_mb_x, i_mb_y, i_num_mb, i_satd_cost;
	h261_mb_mode_decision_t *ps_md_mb;
	me_gpu_mb_t *ps_gpu_mb;

	Int32 i_mb_type;

	i_num_mb = ps_md->i_mb_height * ps_md->i_mb_width;
	i_satd_cost = 0;

	for( i_mb_y = 0; i_mb_y < ps_md->i_mb_height; i_mb_y++ )
	{
		for( i_mb_x = 0; i_mb_x < ps_md->i_mb_width; i_mb_x++ )
		{
			ps_md_mb = &ps_md->prgs_mb[ i_mb_y * ps_md->i_mb_width + i_mb_x ];
			ps_gpu_mb = &ps_me_gpu->prgs_macroblocks_result[ i_mb_y * ps_md->i_mb_width + i_mb_x ];
/*
			printf("intra: %d zero: %d inter: %d filter: %d mv: %d/%d\n", ps_gpu_mb->i_intra_cost,
				ps_gpu_mb->i_zero_cost, ps_gpu_mb->i_16x16_cost, ps_gpu_mb->i_16x16_filter_cost,
				ps_gpu_mb->s_16x16_mv.i8_mx, ps_gpu_mb->s_16x16_mv.i8_my );
*/
			if( ps_picture->i_frame_type == H261_FRAME_TYPE_INTRA )
			{
				i_mb_type = H261_MB_TYPE_INTRA;
			}
			else
			{
				if( ps_gpu_mb->i_intra_cost <= ps_gpu_mb->i_16x16_cost && ps_gpu_mb->i_intra_cost <= ps_gpu_mb->i_16x16_filter_cost  )
				{
					i_mb_type = H261_MB_TYPE_INTRA;
				}
				else
				{
					i_mb_type = H261_MB_TYPE_INTER;
				}
			}
			if( i_mb_type == H261_MB_TYPE_INTRA )
			{
				ps_md_mb->i_mb_type = H261_MB_TYPE_INTRA;
				ps_md_mb->i_cost = ps_gpu_mb->i_intra_cost;
				ps_md_mb->i_mb_flags = 0; /* coeff gets set in encode_macroblock */
				ps_md_mb->i_mv_x = ps_md_mb->i_mv_y = 0;
			}
			else
			{
				ps_md_mb->i_mb_type = H261_MB_TYPE_INTER;
				if( ps_gpu_mb->i_16x16_filter_cost < ps_gpu_mb->i_16x16_cost )
				{
					ps_md_mb->i_mb_flags = H261_MB_MC | H261_MB_FILTER;
					ps_md_mb->i_cost = ps_gpu_mb->i_16x16_filter_cost;
				} 
				else
				{
					ps_md_mb->i_mb_flags = 0;
					ps_md_mb->i_cost = ps_gpu_mb->i_16x16_cost;
				}

				ps_md_mb->i_mv_x = ps_gpu_mb->s_16x16_mv.i8_mx;
				ps_md_mb->i_mv_y = ps_gpu_mb->s_16x16_mv.i8_my;

				if( ps_md_mb->i_mv_x != 0 || ps_md_mb->i_mv_y != 0 )
				{
					ps_md_mb->i_mb_flags |= H261_MB_MC;
				}

				i_satd_cost += ps_md_mb->i_cost;
			}
		}
	}

	/* quant adjust */
	i_satd_cost /= i_num_mb;
	i_satd_cost += 1;

	/*
	for( i_mb_y = 0; i_mb_y < ps_md->i_mb_height; i_mb_y++ )
	{
		for( i_mb_x = 0; i_mb_x < ps_md->i_mb_width; i_mb_x++ )
		{
			Int32 i_satd_ratio;
			Int32 i_quant_adjust_tab[ 11 ] = { -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 };

			ps_md_mb = &ps_md->prgs_mb[ i_mb_y * ps_md->i_mb_width + i_mb_x ];

			i_satd_ratio = ( ( ps_md_mb->i_cost - i_satd_cost ) * 5 ) / i_satd_cost;
			i_satd_ratio += 5;
			i_satd_ratio = MIN( 10, MAX( 0, i_satd_ratio ) );
			ps_md_mb->i_quantiser_adjust = i_quant_adjust_tab[ i_satd_ratio ];
		}
	}
	*/
}
