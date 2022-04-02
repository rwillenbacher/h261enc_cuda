
#define H261_FRAME_TYPE_INTRA				0
#define H261_FRAME_TYPE_INTER				1

#define H261_PICTURE_START_CODE				16
#define H261_PICTURE_START_CODE_LENGTH		20

#define H261_SOURCE_FORMAT_QCIF 0
#define H261_SOURCE_FORMAT_CIF 1

#define H261_GOB_WIDTH		176
#define H261_GOB_HEIGHT		48

#define H261_MB_TYPE_SKIP					-1
#define H261_MB_TYPE_INTRA					0
#define H261_MB_TYPE_INTRA_MQUANT			1
#define H261_MB_TYPE_INTER					2
#define H261_MB_TYPE_INTER_MQUANT			3
#define H261_MB_TYPE_INTER_MC				4
#define H261_MB_TYPE_INTER_MC_COEFF			5
#define H261_MB_TYPE_INTER_MQUANT_MC_COEFF	6
#define H261_MB_TYPE_INTER_MC_FILTER		7
#define H261_MB_TYPE_INTER_MC_FILTER_COEFF	8
#define H261_MB_TYPE_INTER_MQUANT_MC_FILTER_COEFF 9


#define H261_MB_MC	   1
#define H261_MB_FILTER 2
#define H261_MB_COEFF  4


#define H261_CODED_BLOCK_PATTERN_1	32
#define H261_CODED_BLOCK_PATTERN_2	16
#define H261_CODED_BLOCK_PATTERN_3	8
#define H261_CODED_BLOCK_PATTERN_4	4
#define H261_CODED_BLOCK_PATTERN_5	2
#define H261_CODED_BLOCK_PATTERN_6	1


#define H261_RATECTRL_MODE_QUANT	1
#define H261_RATECTRL_MODE_CBR		2


/* motion estimation structures */

typedef struct {
	Int8 i8_mx;
	Int8 i8_my;
} me_gpu_mv_t;

typedef struct {
	Int32 i_intra_cost;
	Int32 i_zero_cost;

	Int32 i_16x16_cost;
	Int32 i_16x16_filter_cost;
	me_gpu_mv_t s_16x16_mv;
} me_gpu_mb_t;


typedef struct me_gpu_s {
	Int32 i_mb_width;
	Int32 i_mb_height;
	Int32 i_num_mb;

	me_gpu_mb_t *prgs_macroblocks_device;
	me_gpu_mb_t *prgs_macroblocks_result;

	me_gpu_mv_t *ps_candidate_vector_device;
	me_gpu_mv_t *ps_candidate_vector_host;

	me_gpu_mv_t *ps_starting_vector_device;
	me_gpu_mv_t *ps_starting_vector_host;

	Int8 *pi8_motion_vector_limits_device;
	Int8 *pi8_motion_vector_limits_host;

	Void *p_cuda;
} me_gpu_t;


/* mode decision structures */

typedef struct {
	Int32 i_cost;		/* satd cost */
	Int32 i_quantiser_adjust; /* +/- quantiser */
	Int32 i_mb_type; /* intra or inter */
	Int32 i_mb_flags; /* same as h261_macroblock_t's */
	Int32 i_mv_x, i_mv_y; /* motion vector */
} h261_mb_mode_decision_t;

typedef struct {
	Int32 i_mb_width;
	Int32 i_mb_height;
	h261_mb_mode_decision_t *prgs_mb;
} h261_mode_decision_t;

/* h261 encoder structures */

typedef struct {
	Int32	i_width;
	Int32	i_height;
	Int32	i_stride_y;
	Int32	i_stride_c;

	UInt8	*pui8_Y;
	UInt8	*pui8_Cr;
	UInt8	*pui8_Cb;
} h261_frame_t;


typedef struct {
	UInt8 *pui8_bitstream;
	Int32 i_length;
	
	Int32 i_byte_count;
	Int32 i_next_bit;

} h261_bitstream_t;


typedef struct {
	Int32 i_denoise;

	Int32 i_intra_count;
	Int32 rgi_intra_noise[ 64 ];
	Int32 rgi_intra_signal_offset[ 64 ];

	Int32 i_inter_count;
	Int32 rgi_inter_noise[ 64 ];
	Int32 rgi_inter_signal_offset[ 64 ];
} h261_dct_denoise_t;


typedef struct h261_macroblock_s {
	/* syntax elements */
	Int32	i_macroblock_address;
	Int32	i_macroblock_type;		/* intra or inter */
	Int32	i_macroblock_type_flags;/* flags of present bitstream elements */

	Int32	i_macroblock_quant;
	Int32	rgi_mv[ 2 ];
	Int32	i_coded_block_pattern;
	Int32	rgi_tcoeff[ 6 ][ 64 ];

	/* data elements */
	Int32	i_mb_x;		/* in pel coords */
	Int32	i_mb_y;
} h261_macroblock_t;


typedef struct {
	Int32 i_mb_width;
	Int32 i_mb_height;
	Int32 i_num_mb;

	h261_macroblock_t *ps_macroblocks;
	h261_macroblock_t *ps_macroblocks_device;

	h261_dct_denoise_t *ps_denoise_host;
	h261_dct_denoise_t *ps_denoise_device;

	Int32 *pi_mb_types_host;
	Int32 *pi_mb_types_device;
	Int32 *pi_mb_flags_host;
	Int32 *pi_mb_flags_device;
	Int32 *pi_mb_quant_host;
	Int32 *pi_mb_quant_device;
	Int32 *pi_mb_mv_x_host;
	Int32 *pi_mb_mv_x_device;
	Int32 *pi_mb_mv_y_host;
	Int32 *pi_mb_mv_y_device;

	UInt8 *pui8_reconstructed_Y_device;
	UInt8 *pui8_reconstructed_Y_host;
	UInt8 *pui8_reconstructed_Cr_device;
	UInt8 *pui8_reconstructed_Cr_host;
	UInt8 *pui8_reconstructed_Cb_device;
	UInt8 *pui8_reconstructed_Cb_host;

#define COEFF_BLOCK_WIDTH  16
#define COEFF_BLOCK_STRIDE_X ( COEFF_BLOCK_WIDTH )
#define COEFF_BLOCK_STRIDE_Y ( COEFF_BLOCK_STRIDE_X * 8 )
#define COEFF_BLOCK_SIZE ( COEFF_BLOCK_STRIDE_Y * 8 )

	Int32 i_num_blocks;
	Int32 i_num_coeff_blocks;
	Int32 *pi_coeff_blocks_device;
	Int32 *pi_coeff_blocks_host;

	Void *p_cuda;
} h261_macroblocks_t;


typedef struct h261_mb_coding_cache_t {
	Int32 i_quantiser;
	Int32 i_last_mv_valid;
	Int32 i_last_mv_x;
	Int32 i_last_mv_y;
} h261_mb_coding_vars_t;


typedef struct {
	/* syntax elements */
	Int32	i_group_of_blocks_start_code; /* 16 bits, 0x0001 */
	Int32	i_group_number;
	Int32	i_quantiser;

	Int32	i_num_extra_insertion_information; /* always 0 */

	/* coding control */
	h261_mb_coding_vars_t s_coding_vars;
} h261_gob_t;


typedef struct {

	/* syntax elements */
	Int32	i_picture_start_code; /* 20 bits, 0x00010 */
	Int32	i_temporal_reference; /* continuity counter */

	/* ptype syntax elements */
	Int32	i_split_screen_indicator;
	Int32	i_document_camera_indicator;
	Int32	i_freeze_picture_release;
	Int32	i_source_format;
	Int32	i_highres_mode;
	Int32	i_spare_bit;	/* always 1 */

	Int32 i_num_extra_insertion_information; /* always 0 */
	UInt8 rgui8_extra_insertion_information[ 1 ];

	/* data elements */
	h261_gob_t		s_groups_of_blocks;

	/* coding control */
	Int32 i_quantiser;
	Int32 i_frame_type;

} h261_picture_t;


typedef struct {
	Int32 i_intra_cost;

	Int32 i_inter_cost;
	Int32 i_inter_filter_cost;
	Int32 i_mv_x, i_mv_y;
} h261_me_t;


typedef struct {
	Int32 i_ratectrl_mode;
	Int32 i_current_quant;

	Int32 i_current_picture_cost;

	Int32 i_prediction_cost[ 2 ];
	Int32 i_prediction_size[ 2 ];

	Int32 i_bitrate;
	Int32 i_vbv_size;
	Int32 i_vbv_fill;
} h261_ratectrl_t;


typedef struct {
	Int32					i_source_format;

	Int32					i_frame_counter;
	h261_frame_t			s_current_frame;
	h261_frame_t			s_reference_frame;

	h261_picture_t			s_picture;

	h261_bitstream_t		s_bitstream;

	h261_mode_decision_t	*ps_mode_decision;

	h261_ratectrl_t			s_ratectrl;

	h261_macroblocks_t		*ps_macroblocks;

} h261_context_t;


typedef struct {
	Int32 i_frame_num;
	Int32 i_source_format;
	Int32 i_quantiser;
	Int32 i_frame_type;
} h261_picture_parameters_t;


