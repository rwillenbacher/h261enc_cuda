


Void h261_bitstream_init( h261_bitstream_t *ps_bitstream, Int32 i_length );
Void h261_bitstream_reset( h261_bitstream_t *ps_bitstream );
Void h261_bitstream_deinit( h261_bitstream_t *ps_bitstream );

Void h261_bitstream_write( h261_bitstream_t *ps_bitstream, UInt32 ui_code, UInt32 ui_length );

Void h261_bitstream_get( h261_bitstream_t *ps_bitstream, UInt8 **ppui8_bitstream, UInt32 *ui_length );

