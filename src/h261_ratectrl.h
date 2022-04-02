Void h261_init_ratectrl( h261_context_t *ps_ctx, Int32 i_quant, Int32 i_bitrate, Int32 i_vbv_size, Int32 i_mode );

Void h261_ratectrl_init_picture( h261_context_t *ps_ctx, h261_picture_parameters_t *ps_picture );

Void h261_ratectrl_update_picture( h261_context_t *ps_ctx, Int32 i_bitstream_length );

Void h261_ratectrl_collect_picture_stats( h261_context_t *ps_ctx, h261_picture_parameters_t *ps_picture );
