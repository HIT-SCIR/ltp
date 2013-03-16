#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <iostream>
#include <iomanip>
#include "Util.h"

class Timer
{
 friend std::ostream& operator<<(std::ostream& os, Timer& t);

 private:
  bool running;
  time_t start_time;

	//TODO in seconds?
  double elapsed_time();

 public:
  /***
   * 'running' is initially false.  A timer needs to be explicitly started
   * using 'start' or 'restart'
   */
  Timer() : running(false), start_time(0) { }

  void start(const char* msg = 0);
//  void restart(const char* msg = 0);
//  void stop(const char* msg = 0);
  void check(const char* msg = 0);

};

/***
 * Return the total time that the timer has been in the "running"
 * state since it was first "started" or last "restarted".  For
 * "short" time periods (less than an hour), the actual cpu time
 * used is reported instead of the elapsed time.
 */
inline double Timer::elapsed_time()
{
	time_t now;
	time(&now);
  return difftime(now, start_time);
}

/***
 * Start a timer.  If it is already running, let it continue running.
 * Print an optional message.
 */
inline void Timer::start(const char* msg)
{
  // Print an optional message, something like "Starting timer t";
	if (msg) TRACE_ERR( msg << std::endl);

  // Return immediately if the timer is already running
  if (running) return;

  // Change timer status to running
  running = true;

  // Set the start time;
  time(&start_time);
}

/***
 * Turn the timer off and start it again from 0.  Print an optional message.
 */
/*
inline void Timer::restart(const char* msg)
{
  // Print an optional message, something like "Restarting timer t";
  if (msg) TRACE_ERR( msg << std::endl;

  // Set the timer status to running
  running = true;

  // Set the accumulated time to 0 and the start time to now
  acc_time = 0;
  start_clock = clock();
  start_time = time(0);
}
*/

/***
 * Stop the timer and print an optional message.
 */
/*
inline void Timer::stop(const char* msg)
{
  // Print an optional message, something like "Stopping timer t";
  check(msg);

  // Recalculate and store the total accumulated time up until now
  if (running) acc_time += elapsed_time();

  running = false;
}
*/
/***
 * Print out an optional message followed by the current timer timing.
 */
inline void Timer::check(const char* msg)
{
  // Print an optional message, something like "Checking timer t";
  if (msg) TRACE_ERR( msg << " : ");

//  TRACE_ERR( "[" << std::setiosflags(std::ios::fixed) << std::setprecision(2) << (running ? elapsed_time() : 0) << "] seconds\n");
  TRACE_ERR( "[" << (running ? elapsed_time() : 0) << "] seconds\n");
}

/***
 * Allow timers to be printed to ostreams using the syntax 'os << t'
 * for an ostream 'os' and a timer 't'.  For example, "cout << t" will
 * print out the total amount of time 't' has been "running".
 */
inline std::ostream& operator<<(std::ostream& os, Timer& t)
{
  //os << std::setprecision(2) << std::setiosflags(std::ios::fixed) << (t.running ? t.elapsed_time() : 0);
  os << (t.running ? t.elapsed_time() : 0);
  return os;
}

#endif // TIMER_H
