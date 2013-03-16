//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: timer.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_TIMER_H__
#define CRFPP_TIMER_H__

#include <ctime>
#include <iostream>
#include <string>
#include <limits>

/* COPY FROM Boost::timer */

namespace CRFPP {

  class timer {
  public:
    explicit timer() { start_time_ = std::clock(); }
    void   restart() { start_time_ = std::clock(); }
    double elapsed() const {
      return  static_cast<double>(std::clock() - start_time_) / CLOCKS_PER_SEC;
    }

    double elapsed_max() const {
      return(static_cast<double>(std::numeric_limits<std::clock_t>::max())
             - static_cast<double>(start_time_)) /
        static_cast<double>(CLOCKS_PER_SEC);
    }

    double elapsed_min() const {
      return static_cast<double>(1.0 / CLOCKS_PER_SEC);
    }

  private:
    std::clock_t start_time_;
  };

  class progress_timer : public timer {
  public:
    explicit progress_timer(std::ostream & os = std::cout) : os_(os) {}
    virtual ~progress_timer() {
      std::istream::fmtflags old_flags = os_.setf(std::istream::fixed,
                                                  std::istream::floatfield);
      std::streamsize old_prec = os_.precision(2);
      os_ << elapsed() << " s\n" << std::endl;
      os_.flags(old_flags);
      os_.precision(old_prec);
    }

  private:
    std::ostream & os_;
  };
}

#endif
