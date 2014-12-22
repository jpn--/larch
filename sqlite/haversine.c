#define SQLITE_CORE
#include <sqlite3ext.h>    //boiler-plate that should appear in every loadable extension
SQLITE_EXTENSION_INIT1     //boiler-plate that should appear in every loadable extension

#include <math.h>

/*
** The greatCircleDistance() SQL function returns the great circle distance between two points, in kilometers.
*/
static void greatCircleDistance(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){

	double R = 6371.1;
	double lat1 = sqlite3_value_double(argv[0]) * 0.0174532925;
	double lat2 = sqlite3_value_double(argv[2]) * 0.0174532925;
	double dLat = (sqlite3_value_double(argv[2])-sqlite3_value_double(argv[0])) * 0.0174532925;
	double dLon = (sqlite3_value_double(argv[3])-sqlite3_value_double(argv[1])) * 0.0174532925;

	double a = sin(dLat/2) * sin(dLat/2) +
        	   sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
	double c = 2 * atan2(sqrt(a), sqrt(1-a)); 
	double d = R * c;
  
	sqlite3_result_double(context, d);
}

/* SQLite invokes this routine once when it loads the extension.
** Create new functions, collating sequences, and virtual table
** modules here.  This is usually the only exported symbol in
** the shared library.
*/
int sqlite3_haversine_init(             //boiler-plate that should appear in every loadable extension
  sqlite3 *db,                          //boiler-plate that should appear in every loadable extension
  char **pzErrMsg,                      //boiler-plate that should appear in every loadable extension
  const sqlite3_api_routines *pApi      //boiler-plate that should appear in every loadable extension
){                                      //boiler-plate that should appear in every loadable extension
  SQLITE_EXTENSION_INIT2(pApi)          //boiler-plate that should appear in every loadable extension
  sqlite3_create_function(db, "greatcircle", 4, SQLITE_ANY, 0, greatCircleDistance, 0, 0);
  return 0;                             //boiler-plate that should appear in every loadable extension
}                                       //boiler-plate that should appear in every loadable extension


void sqlite3_haversine_autoinit()
{
	sqlite3_auto_extension( (void(*)(void))sqlite3_haversine_init );
}

