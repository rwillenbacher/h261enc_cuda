

Void h261_init_mode_decision( h261_mode_decision_t **pps_md, Int32 i_mb_width, Int32 i_mb_height );
Void h261_deinit_mode_decision( h261_mode_decision_t *ps_md );

Void h261_mode_decision( h261_mode_decision_t *ps_md, me_gpu_t *ps_me_gpu, h261_picture_parameters_t *ps_picture );
