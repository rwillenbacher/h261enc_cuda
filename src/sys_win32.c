
#include <windows.h>
#include "h261_decl.h"


double h261_get_time()
{
	double i_ms;
	DWORD dw_tick_count;
	LARGE_INTEGER li_frequency;

	if( QueryPerformanceFrequency( &li_frequency ) == 0 )
	{
		dw_tick_count = GetTickCount();
		i_ms = ( Int32 ) dw_tick_count;
	}
	else
	{
		LARGE_INTEGER li_counter;
		QueryPerformanceCounter( &li_counter );
		i_ms = ( ( double ) li_counter.QuadPart / ( double ) li_frequency.QuadPart ) * 1000.0;
	}


	return i_ms;
}
