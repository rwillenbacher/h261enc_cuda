
#ifdef __cplusplus
extern "C" {
#endif

/* main */
Void h261_gpu_init( );

/* textures */
Void h261_gpu_device_init_textures( Int32 i_width, Int32 i_height );
Void h261_gpu_device_deinit_textures( );

Void h261_gpu_device_set_current( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );
Void h261_gpu_device_set_current_chroma_cb( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );
Void h261_gpu_device_set_current_chroma_cr( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );

Void h261_gpu_device_set_reference( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );
Void h261_gpu_device_set_reference_chroma_cb( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );
Void h261_gpu_device_set_reference_chroma_cr( UInt8 *pui8_frame_data, Int32 i_width, Int32 i_height );


/* motion estimation */
void h261_gpu_device_init_me( me_gpu_t **pps_me_gpu, Int32 i_width, Int32 i_height );
void h261_gpu_device_deinit_me( me_gpu_t **pps_me_gpu );
void h261_gpu_device_me( me_gpu_t *ps_me_gpu );

/* macroblock coder */
Void h261_init_macroblocks( h261_macroblocks_t **pps_mbs, Int32 i_width, Int32 i_height, Int32 i_denoise );
Void h261_deinit_macroblocks( h261_macroblocks_t *ps_mbs );

Void h261_gpu_device_encode_macroblocks( h261_context_t *ps_ctx );
Void h261_gpu_device_decode_macroblocks( h261_context_t *ps_ctx );

#ifdef __cplusplus
}
#endif

