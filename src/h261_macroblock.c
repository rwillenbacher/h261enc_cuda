
#include "h261_decl.h"


Void h261_write_tcoeff( h261_context_t *ps_ctx, Int32 *pi_coeff, Int32 i_is_intra )
{
	Int32 i_idx, i_run, i_level, i_sign, i_ulevel, i_start;

	h261_bitstream_t *ps_bitstream;
	h261_coeff_run_level_t *ps_coeff_run_level;

	ps_bitstream = &ps_ctx->s_bitstream; 

	if( i_is_intra )
	{
		h261_bitstream_write( ps_bitstream, pi_coeff[ 0 ], 8 );
		i_run = 0;
		i_start = 1;
	}
	else
	{
		i_level = pi_coeff[ 0 ];
		if( i_level )
		{
			i_sign = i_level < 0 ? 1 : 0;
			i_ulevel = i_level < 0 ? -i_level : i_level;
			if( i_ulevel == 1 )
			{
				h261_bitstream_write( ps_bitstream, 1, 1 );
				h261_bitstream_write( ps_bitstream, i_sign, 1 );
				i_start = 1;
			}
			else
			{
				i_start = 0;
			}
			i_run = 0;
		}
		else
		{
			i_run = 1;
			i_start = 1;
		}
	}

	for( i_idx = i_start; i_idx < 64; i_idx++ )
	{
		i_level = pi_coeff[ rgi_transmission_order_table[ i_idx ] ];

		if( i_level != 0 )
		{
			i_sign = i_level < 0 ? 1 : 0;
			i_ulevel = i_level < 0 ? -i_level : i_level;

			ps_coeff_run_level = &h261_coeff_run_level_table[ 0 ];
			while( ps_coeff_run_level->i_run != H261_COEFF_RUN_LEVEL_ESCAPE )
			{
				if( ps_coeff_run_level->i_run == i_run && ps_coeff_run_level->i_level == i_ulevel )
				{
					break;
				}
				ps_coeff_run_level++;
			}

			if( ps_coeff_run_level->i_run == H261_COEFF_RUN_LEVEL_ESCAPE )
			{
				h261_bitstream_write( ps_bitstream, 1, 6 );
				h261_bitstream_write( ps_bitstream, i_run, 6 );
				h261_bitstream_write( ps_bitstream, i_level, 8 );
			}
			else
			{
				h261_write_vlc( ps_ctx, &ps_coeff_run_level->s_vlc );
				h261_bitstream_write( ps_bitstream, i_sign, 1 );
			}

			i_run = 0;
		}
		else
		{
			i_run = i_run + 1;
		}
	}

	if( i_run == 64 )
	{
		Int32 *pi_null = 0;
		*pi_null = 0;
	}

	h261_bitstream_write( ps_bitstream, 2, 2 );
}

Void h261_write_macroblock_intra( h261_context_t *ps_ctx, h261_macroblock_t *ps_mb, Int32 i_mb_addr )
{
	Int32 i_idx;

	h261_picture_t *ps_picture;
	h261_gob_t *ps_gob;
	h261_mb_coding_vars_t *ps_coding_vars;
	h261_bitstream_t *ps_bitstream;

	ps_picture = &ps_ctx->s_picture;
	ps_gob = &ps_picture->s_groups_of_blocks;
	ps_coding_vars = &ps_gob->s_coding_vars;
	ps_bitstream = &ps_ctx->s_bitstream; 

	h261_write_vlc( ps_ctx, &h261_mba_table[ i_mb_addr ] );

	if( ps_mb->i_macroblock_quant != ps_coding_vars->i_quantiser )
	{
		h261_write_vlc( ps_ctx, &h261_mtype_table[ H261_MB_TYPE_INTRA_MQUANT ] );
		h261_bitstream_write( ps_bitstream, ps_mb->i_macroblock_quant, 5 );
	}
	else
	{
		h261_write_vlc( ps_ctx, &h261_mtype_table[ H261_MB_TYPE_INTRA ] );
	}

	for( i_idx = 0; i_idx < 6; i_idx++ )
	{
		h261_write_tcoeff( ps_ctx, &ps_mb->rgi_tcoeff[ i_idx ][ 0 ], 1 );
	}
}


Void h261_write_macroblock_inter( h261_context_t *ps_ctx, h261_macroblock_t *ps_mb, Int32 i_mb_addr )
{
	Int32 i_idx, i_coded_quant;
	Int32 i_mb_type;

	const Int32 rgi_type_table[ 2 ][ 8 ] = {
		{
			-1,								/* 0, never */
			H261_MB_TYPE_INTER_MC,			/* 1 */
			-1,								/* 2, never */
			H261_MB_TYPE_INTER_MC_FILTER,	/* 3 */
			H261_MB_TYPE_INTER,				/* 4 */
			H261_MB_TYPE_INTER_MC_COEFF,	/* 5 */
			-1,								/* 6, never */
			H261_MB_TYPE_INTER_MC_FILTER_COEFF, /* 7 */
		},
		{
			-1,								/* 0, never */
			-1,								/* 1, never */
			-1,								/* 2, never */
			-1,								/* 3, never */
			H261_MB_TYPE_INTER_MQUANT,		/* 4 */
			H261_MB_TYPE_INTER_MQUANT_MC_COEFF, /* 5 */
			-1,								/* 6, never */
			H261_MB_TYPE_INTER_MQUANT_MC_FILTER_COEFF /* 7 */
		}
	};

	const Int32 rgi_coeff_table[ 6 ] = {
		H261_CODED_BLOCK_PATTERN_1,
		H261_CODED_BLOCK_PATTERN_2,
		H261_CODED_BLOCK_PATTERN_3,
		H261_CODED_BLOCK_PATTERN_4,
		H261_CODED_BLOCK_PATTERN_5,
		H261_CODED_BLOCK_PATTERN_6
	};

	h261_picture_t *ps_picture;
	h261_gob_t *ps_gob;
	h261_mb_coding_vars_t *ps_coding_vars;
	h261_bitstream_t *ps_bitstream;

	ps_picture = &ps_ctx->s_picture;
	ps_gob = &ps_picture->s_groups_of_blocks;
	ps_coding_vars = &ps_gob->s_coding_vars;
	ps_bitstream = &ps_ctx->s_bitstream; 

	h261_write_vlc( ps_ctx, &h261_mba_table[ i_mb_addr ] );

	if( ps_mb->i_macroblock_quant != ps_coding_vars->i_quantiser &&
		ps_mb->i_macroblock_type_flags & H261_MB_COEFF )
	{
		i_coded_quant = 1;
	}
	else
	{
		i_coded_quant = 0;
	}

	i_mb_type = rgi_type_table[ i_coded_quant ][ ps_mb->i_macroblock_type_flags ];

	if( i_mb_type == -1 )
	{
		Int32 *pi_null = 0;
		*pi_null = 0;
	}

	h261_write_vlc( ps_ctx, &h261_mtype_table[ i_mb_type ] );
	if( i_coded_quant )
	{
		h261_bitstream_write( ps_bitstream, ps_mb->i_macroblock_quant, 5 );
	}


	if( ps_mb->i_macroblock_type_flags & H261_MB_MC )
	{
		Int32 i_mv_x, i_mv_y;
		if( ps_coding_vars->i_last_mv_valid )
		{
			i_mv_x = ps_mb->rgi_mv[ 0 ] - ps_coding_vars->i_last_mv_x;
			i_mv_y = ps_mb->rgi_mv[ 1 ] - ps_coding_vars->i_last_mv_y;
		}
		else
		{
			i_mv_x = ps_mb->rgi_mv[ 0 ];
			i_mv_y = ps_mb->rgi_mv[ 1 ];
		}
		i_mv_x += H261_MVD_TRANSLATION_TABLE_OFFSET;
		i_mv_y += H261_MVD_TRANSLATION_TABLE_OFFSET;
		i_mv_x = rgi_mvd_translation_table[ i_mv_x ];
		i_mv_y = rgi_mvd_translation_table[ i_mv_y ];

		if( i_mv_x < 0 || i_mv_y < 0 )
		{
			Int32 *pi_null = 0;
			*pi_null = 0;
		}
		h261_write_vlc( ps_ctx, &h261_mvd_table[ i_mv_x ] );
		h261_write_vlc( ps_ctx, &h261_mvd_table[ i_mv_y ] );
	}

	if( ps_mb->i_macroblock_type_flags & H261_MB_COEFF )
	{
		if( !ps_mb->i_coded_block_pattern )
		{
			Int32 *pi_null = 0;
			*pi_null = 0;
		}
		h261_write_vlc( ps_ctx, &h261_cbp_table[ ps_mb->i_coded_block_pattern ] );

		for( i_idx = 0; i_idx < 6; i_idx++ )
		{
			if( ps_mb->i_coded_block_pattern & rgi_coeff_table[ i_idx ] )
			{
				h261_write_tcoeff( ps_ctx, &ps_mb->rgi_tcoeff[ i_idx ][ 0 ], 0 );
			}
		}
	}
}




Void h261_write_macroblock( h261_context_t *ps_ctx, h261_macroblock_t *ps_mb, Int32 i_mb_addr )
{
	switch( ps_mb->i_macroblock_type )
	{
	case H261_MB_TYPE_INTRA:
		h261_write_macroblock_intra( ps_ctx, ps_mb, i_mb_addr );
		break;

	case H261_MB_TYPE_INTER:
		h261_write_macroblock_inter( ps_ctx, ps_mb, i_mb_addr );
		break;
	}
}

