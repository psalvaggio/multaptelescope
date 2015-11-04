// An object to store command history for a CLI.
// Author: Philip Salvaggio

#ifndef COMMAND_HIST_H
#define COMMAND_HIST_H

#include <list>
#include <string>

namespace mats_io {

class CommandHistory {
 public:
  // Constructor
  //
  // Arguments:
  //  capacity  The number of commands to remember
  explicit CommandHistory(size_t capacity);

  virtual ~CommandHistory();

  // Adds a command to the history
  //
  // Arguments:
  //  command  The command to add
  void AddCommand(const std::string& command);

  // Commands in the history are accessed using 
  const std::string& operator*() const;
  void operator++();
  void operator--();

  void ResetIterator();

  void PrintHistory() const;

 private:
  size_t capacity_;
  std::list<std::string> history_;
  decltype(std::begin(history_)) current_cmd_;
  const static std::string kBlank;
};

}

#endif
