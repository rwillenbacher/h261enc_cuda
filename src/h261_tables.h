

typedef struct {
	Int32 i_code;
	Int32 i_length;
} h261_vlc_t;


#define H261_COEFF_RUN_LEVEL_ESCAPE -1

typedef struct {
	Int32 i_run;
	Int32 i_level;
	h261_vlc_t s_vlc;
} h261_coeff_run_level_t;


/* macroblock address vlc table */
#define H261_MBA_START_CODE 0
#define H261_MBA_STUFFING   34
extern h261_vlc_t h261_mba_table[];

/* macroblock type vlc table */
extern h261_vlc_t h261_mtype_table[ 10 ];

/* coded block pattern vlc table */
extern h261_vlc_t h261_cbp_table[ 64 ];

/* motion vector difference vlc table */
extern h261_vlc_t h261_mvd_table[ 32 ];

/* run / level vlc table */
extern h261_coeff_run_level_t h261_coeff_run_level_table[ ];

/* motion vector difference lookup table */
#define H261_MVD_TRANSLATION_TABLE_OFFSET 32
extern Int32 rgi_mvd_translation_table[ 64 ];

/* zig zag lookup table */
extern Int32 rgi_transmission_order_table[ 64 ];


Void h261_write_vlc( h261_context_t *ps_ctx, h261_vlc_t *ps_vlc );
