// A framework for a shell-based command line application.
// Author: Philip Salvaggio

#include <csignal>
#include <memory>
#include "command_history.h"

#ifndef SHELL_H
#define SHELL_H

namespace mats_io {

template<typename Delegate>
class Shell {
 public:
  // Constructor
  // 
  // Arguments:
  //  delegate              The shell's delegate. The shell assumes ownership.
  //  cmd_history_capacity  The number of commands to store in history.
  Shell(std::unique_ptr<Delegate> delegate, int cmd_history_capacity=256);

  virtual ~Shell();

  // Runs the shell
  void execute();

  void SetPrompt(const std::string& prompt) { prompt_ = prompt; }

 private:
  // Read a single character as it is typed from stdin
  //
  // Arguments:
  //   echo  If true, will print the character the user types
  //
  // I got this termios solution from:
  // http://www.mombu.com/programming/c/t-checking-for-a-keypress-on-linux--6004322.html
  char getch(bool echo=true);

  // Clears the command buffer.
  void resetBuffer();

  // Prints the prompt 
  void printPrompt();
    
  // Prints the command buffer
  void printBuffer();

  // Clears the current line in the shell
  void clearPrompt();

  // Helper function to handle the action when an arrow key is pressed.
  void arrowHandler();

 private:
  std::unique_ptr<Delegate> delegate_;
  CommandHistory command_history_;
  size_t buffer_pos_;
  std::string buffer_;
  std::string prompt_;
};

}

//void ShellSignalHandler(int sig_num);

#include "shell.hpp"

#endif
