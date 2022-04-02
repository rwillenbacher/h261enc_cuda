
#include "h261_decl.h"

#define H261_RATECTRL_BITS 6
#define H261_RATECTRL_ONE ( 1 << H261_RATECTRL_BITS )
#define H261_RATECTRL_HALF ( 1 << ( H261_RATECTRL_BITS - 1 ) )
/* 29.97 */
#define H261_RATECTRL_TIME_SCALE 1918

/* pow( 2.0, f/5 ), f = -31 to 31 */
static Int32 h261_ratectrl_qscale[ 63 ] = {
	0, 1, 1, 1, 1, 1, 2, 2, 2, 3, /* -31 - -22 */
	3, 4, 4, 5, 6, 6, 8, 9, 10, 12, /* -21 - -12 */
	13, 16, 18, 21, 24, 27, 32, 36, 42, 48, 55,  /* -11 - -1 */
	64, /* 0 */
	73, 84, 97, 111, 128, 147, 168, 194, 222, 256, 294, /* 1 - 11 */
	337, 388, 445, 512, 588, 675, 776, 891, 1024, 1176, /* 12 - 21 */
	1351, 1552, 1782, 2048, 2352, 2702, 3104, 3565, 4096, 4705 /* 22 - 31 */
};


Int32 h261_ratectrl_mul( Int32 m1, Int32 m2 )
{
	Int64 m;

	m = ( ( Int64 ) m1 ) * ( ( Int64 ) m2 );

	return ( Int32 ) ( m >> H261_RATECTRL_BITS );
}


Int32 h261_ratectrl_div( Int32 i_div1, Int32 i_div2 )
{
	Int64 d;

	d = ( ( ( Int64 ) i_div1 ) << H261_RATECTRL_BITS ) / i_div2;

	return ( Int32 ) d;
}


Int32 h261_ratectrl_int_to_fp( Int32 i_int )
{
	return i_int << H261_RATECTRL_BITS;
}


Int32 h261_ratectrl_fp_to_int( Int32 i_fp )
{
	return i_fp >> H261_RATECTRL_BITS;
}


Int32 h261_ratectrl_qscale_for_quant( Int32 i_quant )
{
	i_quant = MIN( MAX( i_quant, 0 ), 31 );
	i_quant += 31;

	return h261_ratectrl_qscale[ i_quant ];
}


Int32 h261_ratectrl_quant_for_qscale( Int32 i_qscale )
{
	Int32 i_pos, i_step, i_lower_idx, i_upper_idx;

	i_pos = 31;
	i_step = 16;

	while( i_step )
	{
		if( h261_ratectrl_qscale[ i_pos ] < i_qscale )
		{
			i_pos += i_step;
		}
		else
		{
			i_pos -= i_step;
		}
		i_step = i_step >> 1;
	}

	if( h261_ratectrl_qscale[ i_pos ] < i_qscale )
	{
		i_lower_idx = i_pos;
		i_upper_idx = MIN( i_pos + 1, 62 );
	}
	else
	{
		i_lower_idx = MAX( 0, i_pos - 1 );
		i_upper_idx = i_pos;
	}

	if( i_qscale - h261_ratectrl_qscale[ i_lower_idx ] < h261_ratectrl_qscale[ i_upper_idx ] - i_qscale )
	{
		i_pos = i_lower_idx;
	}
	else
	{
		i_pos = i_upper_idx;
	}

	return i_pos - 31;
}


Void h261_init_ratectrl( h261_context_t *ps_ctx, Int32 i_quant, Int32 i_bitrate, Int32 i_vbv_size, Int32 i_mode )
{
	Int32 i_idx;

	h261_ratectrl_t *ps_ratectrl;

	ps_ratectrl = &ps_ctx->s_ratectrl;

	ps_ratectrl->i_ratectrl_mode = i_mode;
	ps_ratectrl->i_current_quant = i_quant;

	for( i_idx = 0; i_idx < 2; i_idx++ )
	{
		ps_ratectrl->i_prediction_cost[ i_idx ] = 100;
		ps_ratectrl->i_prediction_size[ i_idx ] = 1;
	}

	ps_ratectrl->i_bitrate = i_bitrate / 8;
	ps_ratectrl->i_vbv_size = i_vbv_size / 8;
	ps_ratectrl->i_vbv_fill = ps_ratectrl->i_vbv_size;
}


Int32 h261_ratectrl_predict_size( h261_ratectrl_t *ps_ratectrl, Int32 i_frame_type, Int32 i_cost, Int32 i_quant )
{
	Int32 i_cost_scale, i_predicted_size, i_size_scale;

	i_cost_scale = h261_ratectrl_div( i_cost, ps_ratectrl->i_prediction_cost[ i_frame_type ] );

	i_predicted_size = h261_ratectrl_mul( i_cost_scale, ps_ratectrl->i_prediction_size[ i_frame_type ] );

	i_size_scale = h261_ratectrl_qscale_for_quant( i_quant );

	return h261_ratectrl_div( i_predicted_size, i_size_scale );
}


Void h261_ratectrl_init_picture( h261_context_t *ps_ctx, h261_picture_parameters_t *ps_picture )
{
	Int32 i_qscale, i_qscale_scale, i_predicted_size, i_frame_type, i_target_size, i_quant_delta;
	Int32 i_quant_vbv_penalty, i_up_scale, i_target_upper, i_target_lower;
	h261_ratectrl_t *ps_ratectrl;

	ps_ratectrl = &ps_ctx->s_ratectrl;

	i_frame_type = ps_picture->i_frame_type;

	/* buffer model */
	ps_ratectrl->i_vbv_fill += h261_ratectrl_div( ps_ratectrl->i_bitrate, H261_RATECTRL_TIME_SCALE );
	ps_ratectrl->i_vbv_fill = MIN( ps_ratectrl->i_vbv_fill, ps_ratectrl->i_vbv_size );

	/* quant adjust */
	i_qscale = h261_ratectrl_qscale_for_quant( ps_picture->i_quantiser );

	i_predicted_size = h261_ratectrl_predict_size( ps_ratectrl, i_frame_type,
		ps_ratectrl->i_current_picture_cost, ps_ratectrl->i_current_quant );

	if( i_frame_type == H261_FRAME_TYPE_INTRA )
	{
		i_target_size = h261_ratectrl_div( ps_ratectrl->i_bitrate, H261_RATECTRL_TIME_SCALE );
		i_target_size *= 20;
	}
	else
	{
		i_target_size = h261_ratectrl_div( ps_ratectrl->i_bitrate, H261_RATECTRL_TIME_SCALE );
	}

	i_up_scale = h261_ratectrl_div( ps_ratectrl->i_vbv_fill, ps_ratectrl->i_vbv_size );
	i_target_upper = i_target_size + h261_ratectrl_mul( i_target_size, i_up_scale );
	i_target_lower = i_target_size - h261_ratectrl_mul( i_target_size, H261_RATECTRL_ONE - i_up_scale );
	i_target_size = MAX( MIN( i_predicted_size, i_target_upper ), i_target_lower );
	i_target_size = MAX( i_target_lower, 10 ); /* 10 bytes min per frame */

	i_qscale_scale = h261_ratectrl_div( i_predicted_size, i_target_size );

	if( i_qscale_scale < H261_RATECTRL_ONE )
	{
		i_qscale_scale = h261_ratectrl_mul( H261_RATECTRL_ONE - i_up_scale, H261_RATECTRL_ONE ) +
			h261_ratectrl_mul( i_up_scale, i_qscale_scale );
	}
	i_quant_delta = h261_ratectrl_quant_for_qscale( i_qscale_scale );

	ps_ratectrl->i_current_quant += i_quant_delta;
	ps_ratectrl->i_current_quant = MIN( MAX( ps_ratectrl->i_current_quant, 1 ), 31 );

	i_quant_vbv_penalty = h261_ratectrl_fp_to_int( ( H261_RATECTRL_ONE - i_up_scale ) * 4 );

	ps_picture->i_quantiser = ps_ratectrl->i_current_quant + i_quant_vbv_penalty;
}


Void h261_ratectrl_update_picture( h261_context_t *ps_ctx, Int32 i_bitstream_length )
{
	Int32 i_size_scale, i_frame_type;
	h261_picture_t *ps_picture;
	h261_ratectrl_t *ps_ratectrl;

	ps_picture = &ps_ctx->s_picture;
	ps_ratectrl = &ps_ctx->s_ratectrl;

	/* frame size prediction update */
	i_frame_type = ps_picture->i_frame_type;

	ps_ratectrl->i_prediction_cost[ i_frame_type ] =
		h261_ratectrl_mul( ps_ratectrl->i_prediction_cost[ ps_picture->i_frame_type ], H261_RATECTRL_HALF );
	ps_ratectrl->i_prediction_size[ i_frame_type ] =
		h261_ratectrl_mul( ps_ratectrl->i_prediction_size[ i_frame_type ], H261_RATECTRL_HALF );

	ps_ratectrl->i_prediction_cost[ i_frame_type ] += ps_ratectrl->i_current_picture_cost;

	i_size_scale = h261_ratectrl_qscale_for_quant( ps_picture->i_quantiser );
	ps_ratectrl->i_prediction_size[ i_frame_type ] += h261_ratectrl_mul( i_bitstream_length, i_size_scale );

	/* buffer model */
	ps_ratectrl->i_vbv_fill -= i_bitstream_length;
}


Void h261_ratectrl_collect_picture_stats( h261_context_t *ps_ctx, h261_picture_parameters_t *ps_picture )
{
	Int32 i_num_mb, i_idx, i_picture_cost;
	h261_ratectrl_t *ps_ratectrl;
	h261_mode_decision_t *ps_md;
	h261_mb_mode_decision_t *ps_md_mb;

	ps_md = ps_ctx->ps_mode_decision;
	ps_ratectrl = &ps_ctx->s_ratectrl;
	
	i_num_mb = ps_md->i_mb_width * ps_md->i_mb_height;
	i_picture_cost = 0;

	for( i_idx = 0; i_idx < i_num_mb; i_idx++ )
	{
		ps_md_mb = &ps_md->prgs_mb[ i_idx ];
		i_picture_cost += ps_md_mb->i_cost;
	}

	ps_ratectrl->i_current_picture_cost = MAX( i_picture_cost, i_num_mb * 100 );
}



