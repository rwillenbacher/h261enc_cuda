typedef void Void;

typedef char Int8;
typedef unsigned char UInt8;

typedef short Int16;
typedef unsigned short UInt16;

typedef int Int32;
typedef unsigned int UInt32;

typedef long long Int64;
typedef unsigned long long UInt64;

#ifndef MIN
#define MIN( x, y ) ( (x) < (y) ? (x) : (y) )
#endif

#ifndef MAX
#define MAX( x, y ) ( (x) > (y) ? (x) : (y) )
#endif
