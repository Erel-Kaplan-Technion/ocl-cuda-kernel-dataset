// exception.h - minimal replacement for missing NVIDIA SDK header
#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

#include <stdexcept>

#define RUNTIME_EXCEPTION(msg) throw std::runtime_error(msg)
#define LOGIC_EXCEPTION(msg) throw std::logic_error(msg)

#endif // _EXCEPTION_H_
