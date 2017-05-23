#ifndef CALL_TRACE_HPP
#define CALL_TRACE_HPP

#ifndef CALL_TRACE
#ifdef CALLTRACE
#define CALL_TRACE() call_trace_callback(__FUNCTION__,__LINE__,__FILE__)
#else
#define CALL_TRACE()
#endif
#endif

void call_trace_callback(const char* func, int line, const char* file);

#endif
