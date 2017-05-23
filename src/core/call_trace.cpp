#include "call_trace.hpp"
#include <time.h>
#include <unistd.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include "utils.hpp"

#define FOLDED

struct call_t {
  int time;
  bool known;
  char *name;
  void* addr;
  void* ret;
  call_t* next;
  call_t* called;
  int line;
  char *file;
};

call_t* calls = NULL;

void call_trace_write_cb (FILE* h, call_t* c) {
  if (c == NULL) return;
  if (c->known == false) {
    int cnt = 1;
#ifdef FOLDED
    while (c->known == false && c->next == NULL && c->called->known == false) {
      c = c->called;
      ++cnt;
    }
#endif
    if (cnt > 1)
      fprintf(h,"<call name=\"[%i unknown]\" time=\"\" addr=\"\">\n",
        cnt);
    else
      fprintf(h,"<call name=\"%s\" time=\"\" addr=\"0x%016lx\">\n",
        c->name,(size_t)c->ret);
  } else {
    fprintf(h,"<call name=\"%s\" time=\"%i\" addr=\"0x%016lx\" line=\"%i\" file=\"%s\">\n",
      c->name,c->time,(size_t)c->ret,c->line,c->file);
  }
  call_trace_write_cb(h, c->called);
  fprintf(h,"</call>\n");
  call_trace_write_cb(h, c->next);
}

void call_trace_write () {
  char filename[32]; 
  sprintf(filename,"calltrace.out.%i", this_node);//getpid());
  FILE* h = fopen(filename,"w");
  call_trace_write_cb(h, calls);
  fclose(h);
}

void call_trace_callback (const char* func, int line, const char* file) {
  void* stack[32];
  int size;
  char** strings;
  int time = clock();
  size = (int)backtrace(stack, 32);
  strings = backtrace_symbols(stack, size);
    
  call_t** call_ptr = &calls;
  if (calls == NULL) {
    atexit(call_trace_write);
  }
    
  for (int i=size-3; i > 1;--i) {
    if (*call_ptr != NULL) {
      if ((*call_ptr)->addr == stack[i+1]) {
        call_ptr = &(*call_ptr)->called;
      } else {
        call_ptr = &(*call_ptr)->next;
        ++i;
      }
    } else {
      *call_ptr = new call_t();
      (*call_ptr)->time = time;
      (*call_ptr)->name = new char[strlen(strings[i]) + 1];
      sprintf((*call_ptr)->name,"%s",basename(strings[i]));
      (*call_ptr)->addr = stack[i+1];
      (*call_ptr)->ret = stack[i];
      (*call_ptr)->known = false;
      (*call_ptr)->line = 0;
      (*call_ptr)->file = NULL;
      (*call_ptr)->next = NULL;
      (*call_ptr)->called = NULL;
      call_ptr = &(*call_ptr)->called;
    }
  }
    
  while (*call_ptr != NULL)
    call_ptr = &(*call_ptr)->next;
  *call_ptr = new call_t();
  (*call_ptr)->time = time;
  (*call_ptr)->name = new char[strlen(func) + 1];
  sprintf((*call_ptr)->name,"%s",func);
  (*call_ptr)->addr = stack[2];
  (*call_ptr)->ret = stack[1];
  (*call_ptr)->known = true;
  (*call_ptr)->line = line;
  char file_tmp[strlen(file) + 1];
  sprintf(file_tmp,"%s",file);
  (*call_ptr)->file = new char[strlen(file) + 1];
  sprintf((*call_ptr)->file,"%s",basename(file_tmp));
  (*call_ptr)->next = NULL;
  (*call_ptr)->called = NULL;
}
