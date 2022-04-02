
#include "h261_decl.h"



Void h261_init_picture( h261_context_t *ps_ctx, h261_picture_parameters_t *ps_parameters )
{
	h261_picture_t *ps_picture;
	ps_picture = &ps_ctx->s_picture;

	ps_picture->i_temporal_reference = ps_parameters->i_frame_num & 0x1f;

	ps_picture->i_spare_bit = 1;
	ps_picture->i_highres_mode = 1;
	ps_picture->i_source_format = ps_parameters->i_source_format;
	ps_picture->i_freeze_picture_release = 1;
	ps_picture->i_document_camera_indicator = 0;
	ps_picture->i_split_screen_indicator = 0;

	ps_picture->i_num_extra_insertion_information = 0;

	/* coding control */
	ps_picture->i_frame_type = ps_parameters->i_frame_type;

	ps_picture->i_quantiser = ps_parameters->i_quantiser;
	if( ps_picture->i_quantiser < 1 )
	{
		ps_picture->i_quantiser = 1;
	}
	else if( ps_picture->i_quantiser > 31 )
	{
		ps_picture->i_quantiser = 31;
	}

}


Void h261_write_picture_header( h261_context_t *ps_ctx )
{
	Int32 i_idx;
	h261_frame_t *ps_current_frame, *ps_reference_frame;
	h261_picture_t *ps_picture;
	h261_bitstream_t *ps_bitstream;

	ps_current_frame = &ps_ctx->s_current_frame;
	ps_reference_frame = &ps_ctx->s_reference_frame;
	ps_bitstream = &ps_ctx->s_bitstream;
	ps_picture = &ps_ctx->s_picture;

	/* startcode + frame counter */
	h261_bitstream_write( ps_bitstream, H261_PICTURE_START_CODE, H261_PICTURE_START_CODE_LENGTH );
	h261_bitstream_write( ps_bitstream, ps_picture->i_temporal_reference, 5 );

	/* ptype */
	h261_bitstream_write( ps_bitstream, ps_picture->i_split_screen_indicator, 1 );
	h261_bitstream_write( ps_bitstream, ps_picture->i_document_camera_indicator, 1 );
	h261_bitstream_write( ps_bitstream, ps_picture->i_freeze_picture_release, 1 );
	h261_bitstream_write( ps_bitstream, ps_picture->i_source_format, 1 );
	h261_bitstream_write( ps_bitstream, ps_picture->i_highres_mode, 1 );
	h261_bitstream_write( ps_bitstream, ps_picture->i_spare_bit, 1 );

	/* extra insertion information */
	for( i_idx = 0; i_idx < ps_picture->i_num_extra_insertion_information; i_idx++ )
	{
		h261_bitstream_write( ps_bitstream, 1, 1 );
		h261_bitstream_write( ps_bitstream, ps_picture->rgui8_extra_insertion_information[ i_idx ] , 8 );
	}
	h261_bitstream_write( ps_bitstream, 0, 1 );


}














