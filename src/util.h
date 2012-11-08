#pragma once
#include <iostream>
#include <sstream>
#define CUR_LOCATION "@" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ">>"
#define PRINT(msg) std::cout << CUR_LOCATION << " >> " << msg << std::endl
#define ERROR(msg) PRINT(msg); throw msg;
#define HALT(msg) PRINT(msg); exit;
#define COUT(msg) PRINT(msg);


namespace util{
	//use util::cout instead of std::cout to
	//have additional features:
	//
	// (1) counter before the output stream, but only after a newline
	// (2) current location after each newline
	//
		class MyStream: public std::ostream
		{
			class MyStreamBuf: public std::stringbuf
			{
					std::ostream& output;
					public:
							MyStreamBuf(std::ostream& str) :output(str) {}

					virtual int sync ( )
					{
							output << CUR_LOCATION << str();
							str("");
							output.flush();
							return 0;
					}
			};
			MyStreamBuf buffer;
			public:
					MyStream(std::ostream& str) : std::ostream(&buffer) ,buffer(str) {};
		};
		MyStream cout(std::cout);
};

