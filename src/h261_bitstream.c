
#include "h261_decl.h"


Void h261_bitstream_init( h261_bitstream_t *ps_bitstream, Int32 i_length )
{
	ps_bitstream->pui8_bitstream = malloc( sizeof( UInt8 ) * i_length );
	ps_bitstream->i_length = i_length;

	h261_bitstream_reset( ps_bitstream );
}

Void h261_bitstream_deinit( h261_bitstream_t *ps_bitstream )
{
	free( ps_bitstream->pui8_bitstream );
}

Void h261_bitstream_reset( h261_bitstream_t *ps_bitstream )
{
	memset( ps_bitstream->pui8_bitstream, 0, sizeof( UInt8 ) * ps_bitstream->i_length );

	ps_bitstream->i_next_bit = 7;
	ps_bitstream->i_byte_count = 0;

}


Void h261_bitstream_write( h261_bitstream_t *ps_bitstream, UInt32 ui_code, UInt32 ui_length )
{
	Int32 i_bit_pos;
	UInt32 ui_masked_bit, ui_byte_pos;

	i_bit_pos = ps_bitstream->i_next_bit;
	ui_byte_pos = ps_bitstream->i_byte_count;

	ui_masked_bit = 1 << ( ui_length - 1 );

	while( ui_length )
	{
		if( i_bit_pos < 0 )
		{
			i_bit_pos = 7;
			ui_byte_pos++;
		}
		if( ui_code & ui_masked_bit )
		{
			UInt8 ui8_byte;

			ui8_byte = ps_bitstream->pui8_bitstream[ ui_byte_pos ];
			ui8_byte |= 1 << i_bit_pos;
			ps_bitstream->pui8_bitstream[ ui_byte_pos ] = ui8_byte;

		}
		i_bit_pos--;
		ui_masked_bit >>= 1;
		ui_length--;
	}

	ps_bitstream->i_byte_count = ui_byte_pos;
	ps_bitstream->i_next_bit = i_bit_pos;
}


Void h261_bitstream_get( h261_bitstream_t *ps_bitstream, UInt8 **ppui8_bitstream, UInt32 *pui_length )
{
	*ppui8_bitstream = ps_bitstream->pui8_bitstream;

	if( ps_bitstream->i_next_bit != 7 )
	{
		*pui_length = ps_bitstream->i_byte_count + 1;
	}
	else
	{
		*pui_length = ps_bitstream->i_byte_count;
	}
}

