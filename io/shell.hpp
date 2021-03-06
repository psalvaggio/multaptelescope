// Implementation file for the Shell class.
// Author: Philip Salvaggio

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <signal.h>
#include <sys/types.h>

#include "shell.h"

#ifndef SHELL_HPP
#define SHELL_HPP

namespace mats_io {

template<typename Delegate>
Shell<Delegate>::Shell(std::unique_ptr<Delegate> delegate,
                       int cmd_history_capacity)
    : delegate_(std::move(delegate)),
      command_history_(cmd_history_capacity),
      buffer_pos_(0),
      buffer_(), 
      prompt_("Shell> ") {}


template<typename Delegate>
Shell<Delegate>::~Shell() {}


template<typename Delegate>
char Shell<Delegate>::getch(bool echo) {
  char ch;
  int error;
  static struct termios oldTTY, newTTY;

  fflush(stdout);
  tcgetattr(STDIN_FILENO, &oldTTY);
  newTTY = oldTTY;

  newTTY.c_lflag &= ~ICANON;  // line settings

  // Enable/disable echo
  if (echo) newTTY.c_lflag |= ECHO;
  else newTTY.c_lflag &= ~ECHO;

  newTTY.c_cc[VMIN] = 1;   // minimum chars to wait for 
  newTTY.c_cc[VTIME] = 1;  // minimum wait time 

  error = tcsetattr(STDIN_FILENO, TCSANOW, &newTTY);
  if (!error) {
    // get a single character from stdin 
    error = read(STDIN_FILENO, &ch, 1);
        
    // restore old settings 
    error += tcsetattr( STDIN_FILENO, TCSANOW, &oldTTY );
  }

  return error == 1 ? ch : '\0';
}


template<typename Delegate>
void Shell<Delegate>::execute() {
  // Print out the prompt
  printPrompt();

  // Loop forever
  while (true) {
    // Grab the input
    char tmpChar = getch(false);

    // If an error occured (Ctrl+C causes such behavior), just skip this
    // iteration
    if (tmpChar == 0) continue;
    // If the character was Ctrl+D, quit
    else if (tmpChar == 4) {
      std::cout << std::endl;
      return;

    // If the character was a tab, we want to do tab completion
    } else if (tmpChar == '\t') {
      tabCompletion();

    // If the character was a new line, we might want to do something
    } else if (tmpChar == '\n') {
      // Print out the new line
      std::cout << std::endl;

      // If they didn't type anything, give them a new prompt
      if (buffer_pos_ == 0) {
        resetBuffer();
        printPrompt();
        continue;
      }
    
      // Put the command into the command history
      std::string tmp_command = buffer_;
      mats::trim(tmp_command);
      buffer_pos_ = 0;
      resetBuffer();
      command_history_.AddCommand(tmp_command);
    
      // If they typed "exit", we're done here
      if (tmp_command == "exit") return;

      // We need to tokenize the command
      std::vector<std::string> command_parts;
      mats::explode(tmp_command, "[\\s]+", &command_parts);
      delegate_->ProcessCommand(command_parts);
      printPrompt();

    } else if (tmpChar == 27) {  // Arrow keys
      arrowHandler();

    } else if (tmpChar == 8 || tmpChar == 127) { // Backspace
      if (buffer_pos_ > 0) {
        buffer_.erase(buffer_pos_ - 1, 1);
        buffer_pos_--;
        std::cout << "\b" << buffer_.substr(buffer_pos_) << " \b";
        int go_back_by = buffer_.size() - buffer_pos_;
        for (int i = 0; i < go_back_by; i++) {
          std::cout << "\b";
        }
        std::cout.flush();
      }
    } else {    // Echo back all other characters
      if (buffer_pos_ < buffer_.length()) {
        buffer_.insert(buffer_pos_, &tmpChar, 1);
        std::cout << buffer_.substr(buffer_pos_);
        int go_back_by = buffer_.size() - buffer_pos_ - 1;
        for (int i = 0; i < go_back_by; i++) {
          std::cout << "\b";
        }
      } else {
        buffer_.push_back(tmpChar);
        std::cout << tmpChar;
      }
      buffer_pos_++;
      std::cout.flush();
    }
  }
}


template<typename Delegate>
void Shell<Delegate>::resetBuffer() {
  buffer_.clear();
}


template<typename Delegate>
void Shell<Delegate>::printPrompt() {
  std::cout << prompt_;
  std::cout.flush();
}


template<typename Delegate>
void Shell<Delegate>::arrowHandler() {
  //char firstChar = getch(false);
  getch(false);
  char tmpChar = getch(false);

  std::string command;

  if (tmpChar == 65) {  // Up arrow
    ++command_history_;
  } else if (tmpChar == 66) {  // Down arrow
    --command_history_;
  } else if (tmpChar == 67) {  // Right arrow
    if (buffer_pos_ != buffer_.size()) {
      std::cout << buffer_[buffer_pos_];
      std::cout.flush();
      buffer_pos_++;
    }
    return;
  } else if (tmpChar == 68) {  // Left Arrow  
    if (buffer_pos_ > 0) {
      std::cout << "\b";
      std::cout.flush();
      buffer_pos_--;
    }
    return;
  } else {
    return;
  }
  command = *command_history_;
   
  // Set the command in the history to the current command     
  clearPrompt();
  printPrompt();
  std::cout << command;
  std::cout.flush();
  buffer_ = command;
  buffer_pos_ = command.size();
}

template<typename Delegate>
void Shell<Delegate>::clearPrompt() {
  std::cout << "\r" << std::string(80, ' ') << "\r";
  std::cout.flush();
}

template<typename Delegate>
void Shell<Delegate>::tabCompletion() {
  // Grab the path the user is currently typing, breaks on spaces and quotes and
  // equals signs
  int path_start = buffer_pos_;
  while (path_start > 0) {
    char tmp = buffer_[path_start-1];
    if (tmp == '\"' || tmp == '=') break;
    if (tmp == ' ') {
      if (path_start - 2 < 0) break;
      if (buffer_[path_start-2] != '\\') break;
    }
    path_start--;
  }

  // Break the path up into directory name and basename
  std::string path = buffer_.substr(path_start, buffer_pos_ - path_start);
  std::string dir_name = mats::DirectoryName(path);
  std::string basename = mats::Basename(path);

  // Scan for files in the given directory
  std::vector<std::string> filenames;
  mats::scandir(dir_name, "", &filenames);

  // Find the shared prefix of all filenames matching the basename
  int count = 0;
  std::string completion = "";
  for (const auto& fname : filenames) {
    if (mats::starts_with(fname, basename)) {
      count++;
      if (count == 1) {
        completion = fname.substr(basename.length());
      } else {
        std::string other = fname.substr(basename.length());
        for (size_t i = 0; i < completion.size() && i < other.size(); i++) {
          if (completion[i] != other[i]) {
            completion = completion.substr(0, i);
            break;
          }
        }
      }
    }
  }

  // If no completion or conflicting completions were found, beep at the user
  if (count != 1) {
    std::cout << "\a";
    if (completion.empty()) return;
  }

  // If the completion is a directory, add a slash to the end of it
  if (mats::is_dir(path + completion)) completion += "/";

  // Insert the completion and echo it out
  buffer_.insert(buffer_pos_, completion);
  std::cout << buffer_.substr(buffer_pos_);
  buffer_pos_ += completion.length();
  int go_back_by = buffer_.size() - buffer_pos_;
  for (int i = 0; i < go_back_by; i++) {
    std::cout << "\b";
  }
  std::cout.flush();
}

}

#endif
