
#include "h261_decl.h"



Void h261_write_gobs( h261_context_t *ps_ctx )
{
	Int32 i_idx;
	Int32 i_gob_idx, i_offset_x, i_offset_y;

	if( ps_ctx->i_source_format == H261_SOURCE_FORMAT_CIF )
	{
		const Int32 rgi_cif[ 12 ][ 3 ] = {
			{ 1, 0, 0 },
			{ 2, 1, 0 },
			{ 3, 0, 1 },
			{ 4, 1, 1 },
			{ 5, 0, 2 },
			{ 6, 1, 2 },
			{ 7, 0, 3 },
			{ 8, 1, 3 },
			{ 9, 0, 4 },
			{ 10, 1, 4 },
			{ 11, 0, 5 },
			{ 12, 1, 5 }
		};

		for( i_idx = 0; i_idx < 12; i_idx++ )
		{
			i_gob_idx = rgi_cif[ i_idx ][ 0 ];
			i_offset_x = rgi_cif[ i_idx ][ 1 ] * H261_GOB_WIDTH;
			i_offset_y = rgi_cif[ i_idx ][ 2 ] * H261_GOB_HEIGHT;

			h261_write_gob( ps_ctx, i_gob_idx, i_offset_x, i_offset_y );
		}
	}
	else
	{
		const Int32 rgi_qcif[ 3 ][ 3 ] = {
			{ 1, 0, 0 },
			{ 3, 0, 1 },
			{ 5, 0, 2 }
		};

		for( i_idx = 0; i_idx < 3; i_idx++ )
		{
			i_gob_idx = rgi_qcif[ i_idx ][ 0 ];
			i_offset_x = rgi_qcif[ i_idx ][ 1 ] * H261_GOB_WIDTH;
			i_offset_y = rgi_qcif[ i_idx ][ 2 ] * H261_GOB_HEIGHT;

			h261_write_gob( ps_ctx, i_gob_idx, i_offset_x, i_offset_y );
		}
	}
}



Void h261_write_gob( h261_context_t *ps_ctx, Int32 i_gob_index, Int32 i_offset_x, Int32 i_offset_y )
{
	Int32 i_mb_idx, i_mb_addr;
	Int32 i_mb_x, i_mb_y, i_mbs_x, i_mbs_y;

	h261_picture_t *ps_picture;
	h261_gob_t *ps_gob;
	h261_macroblocks_t *ps_mbs;
	h261_macroblock_t *ps_mb;
	h261_mb_coding_vars_t *ps_coding_vars;

	const Int32 rgi_macroblock_offsets[ 33 ][ 2 ] = {
		{ 0, 0 },
		{ 16, 0 },
		{ 32, 0 },
		{ 48, 0 },
		{ 64, 0 },
		{ 80, 0 },
		{ 96, 0 },
		{ 112, 0 },
		{ 128, 0 },
		{ 144, 0 },
		{ 160, 0 },

		{ 0, 16 },
		{ 16, 16 },
		{ 32, 16 },
		{ 48, 16 },
		{ 64, 16 },
		{ 80, 16 },
		{ 96, 16 },
		{ 112, 16 },
		{ 128, 16 },
		{ 144, 16 },
		{ 160, 16 },

		{ 0, 32 },
		{ 16, 32 },
		{ 32, 32 },
		{ 48, 32 },
		{ 64, 32 },
		{ 80, 32 },
		{ 96, 32 },
		{ 112, 32 },
		{ 128, 32 },
		{ 144, 32 },
		{ 160, 32 },
	};

	ps_picture = &ps_ctx->s_picture;
	ps_gob = &ps_picture->s_groups_of_blocks;
	ps_mbs = ps_ctx->ps_macroblocks;
	ps_coding_vars = &ps_gob->s_coding_vars;


	ps_gob->i_group_of_blocks_start_code = 0x0001;
	ps_gob->i_group_number = i_gob_index;
	ps_gob->i_quantiser = ps_picture->i_quantiser;
	ps_gob->i_num_extra_insertion_information = 0;

	ps_coding_vars->i_quantiser = ps_gob->i_quantiser;
	ps_coding_vars->i_last_mv_valid = 0;
	
	h261_write_gob_header( ps_ctx );

	i_mb_addr = 0;
	for( i_mb_idx = 1; i_mb_idx < 34; i_mb_idx++ )
	{
		Int32 i_coded;

		i_mb_addr++;

		i_mb_x = rgi_macroblock_offsets[ i_mb_idx - 1 ][ 0 ] + i_offset_x;
		i_mb_y = rgi_macroblock_offsets[ i_mb_idx - 1 ][ 1 ] + i_offset_y;

		if( i_mb_idx == 1 || i_mb_idx == 12 || i_mb_idx == 23 )
		{
			ps_coding_vars->i_last_mv_valid = 0;
		}

		i_mbs_y = i_mb_y >> 4;
		i_mbs_x = i_mb_x >> 4;
		ps_mb = &ps_mbs->ps_macroblocks[ i_mbs_y * ps_mbs->i_mb_width + i_mbs_x ];

		i_coded = ps_mb->i_macroblock_type_flags;

		if( i_coded )
		{
			h261_write_macroblock( ps_ctx, ps_mb, i_mb_addr );
			i_mb_addr = 0;

			if( ps_mb->i_macroblock_type_flags & H261_MB_COEFF )
			{
				ps_coding_vars->i_quantiser = ps_mb->i_macroblock_quant;
			}

			if( ps_mb->i_macroblock_type_flags & H261_MB_MC )
			{
				ps_coding_vars->i_last_mv_x = ps_mb->rgi_mv[ 0 ];
				ps_coding_vars->i_last_mv_y = ps_mb->rgi_mv[ 1 ];
				ps_coding_vars->i_last_mv_valid = 1;
			}
			else
			{
				ps_coding_vars->i_last_mv_valid = 0;
			}
		}
		else
		{
			ps_coding_vars->i_last_mv_valid = 0;
		}
	}
}





Void h261_write_gob_header( h261_context_t *ps_ctx )
{
	Int32 i_idx;
	h261_bitstream_t *ps_bitstream;
	h261_picture_t *ps_picture;
	h261_gob_t *ps_gob;

	ps_bitstream = &ps_ctx->s_bitstream;

	ps_picture = &ps_ctx->s_picture;
	ps_gob = &ps_picture->s_groups_of_blocks;

	h261_bitstream_write( ps_bitstream, ps_gob->i_group_of_blocks_start_code, 16 );
	h261_bitstream_write( ps_bitstream, ps_gob->i_group_number, 4 );
	h261_bitstream_write( ps_bitstream, ps_gob->i_quantiser, 5 );

	i_idx = 0;
	while( i_idx < ps_gob->i_num_extra_insertion_information )
	{
		h261_bitstream_write( ps_bitstream, 1, 1 );
		h261_bitstream_write( ps_bitstream, 0, 8 );
	}
	h261_bitstream_write( ps_bitstream, 0, 1 );
}