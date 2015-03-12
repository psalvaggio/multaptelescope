// File Description
// Author: Philip Salvaggio

#include "menu_application.h"

#include <curses.h>
#include <menu.h>

using namespace std;

namespace mats {

MenuApplication::MenuItem::MenuItem(const std::string& title,
                   const std::string& description,
                   const action_t& action)
    : title_(title), description_(description), action_(action) {}

MenuApplication::MenuItem::~MenuItem() {}
           
void MenuApplication::MenuItem::GetCursesItem(ITEM** item) {
  *item = new_item(title_.c_str(), description_.c_str());
  set_item_userptr(*item, this);
}


MenuApplication::Menu::Menu(MenuApplication* app,
                            Menu* parent)
    : items_(), app_(app), parent_(parent) {}

MenuApplication::Menu::~Menu() {} 

void MenuApplication::Menu::AddItem(const std::string& title, 
                                    const std::string& description,
                                    const action_t& action) {
  items_.emplace_back(new MenuItem(title, description, action));
}

void MenuApplication::Menu::AddBackItem() {
  items_.emplace_back(new MenuItem("Back...", "", [this]() {
      app_->PopMenu();
  }));
}

MenuApplication::Menu* MenuApplication::Menu::AddSubmenu(
    const std::string& title,
    const std::string& description) {
  Submenu* submenu = new Submenu(title, description, app_, this);
  items_.emplace_back(submenu);
  return submenu;
}

void MenuApplication::Menu::GetCursesMenu(MENU** menu, ITEM*** items) {
  *items = new ITEM*[items_.size() + 1];
  for (size_t i = 0; i < items_.size(); i++) {
    items_[i]->GetCursesItem(&((*items)[i]));
  }
  (*items)[items_.size()] = NULL;
  *menu = new_menu(*items);
}


MenuApplication::Submenu::Submenu(const std::string& title,
                                  const std::string& description,
                                  MenuApplication* app,
                                 MenuApplication::Menu* parent) :
  MenuItem(title, description, [](){}),
  Menu(app, parent) {}


MenuApplication::MenuApplication()
    : keep_going_(true),
      main_menu_(this, nullptr),
      current_menu_(&main_menu_),
      current_curses_items_(nullptr),
      current_curses_menu_(nullptr) {}

MenuApplication::~MenuApplication() {
  FreeCursesMenus();
}

void MenuApplication::AddItem(const std::string& title,
                              const std::string& description,
                              const action_t& action) {
  main_menu_.AddItem(title, description, action);
}

MenuApplication::Menu* MenuApplication::AddSubmenu(
    const std::string& title,
    const std::string& description) {
  return main_menu_.AddSubmenu(title, description);
}

void MenuApplication::stop() {
  keep_going_ = false;
}

void MenuApplication::run() {
  // Curses init
  initscr();
  cbreak();
  timeout(100);
  noecho();
  keypad(stdscr, TRUE);
  raw();
  nonl();

  MakeCursesMenus();
  
  // Present the menu.
  mvprintw(LINES - 2, 0, "'q' to exit");
  post_menu(current_curses_menu_);

  int keycode = 0;
  while (keep_going_ && (keycode = getch()) != 'q') {
    if (keycode == -1) continue;

    switch (keycode) {
      case KEY_DOWN:
        menu_driver(current_curses_menu_, REQ_DOWN_ITEM);
        break;
      case KEY_UP:
        menu_driver(current_curses_menu_, REQ_UP_ITEM);
        break;
      case 13:   // Enter
        MenuItem* item = static_cast<MenuItem*>(
            item_userptr(current_item(current_curses_menu_)));
        if (item->is_submenu() == 0) {
          item->PerformAction();
        } else {
          current_menu_ = static_cast<Menu*>(static_cast<Submenu*>(item));
          MakeCursesMenus();
          post_menu(current_curses_menu_);
        }
        break;
    }
  }

  keep_going_ = false;
  FreeCursesMenus();

  endwin();
}

void MenuApplication::PostStatusMessage(const std::string& message) {
  unique_ptr<char[]> blank_line(new char[80]);
  for (int i = 0; i < 80; i++) (blank_line.get())[i] = ' ';

  mvprintw(LINES - 1, 0, blank_line.get());
  mvprintw(LINES - 1, 0, message.c_str());
}

void MenuApplication::PushMenu(Menu* menu) {
  current_menu_ = menu;
  MakeCursesMenus();
  post_menu(current_curses_menu_);
}

void MenuApplication::PopMenu() {
  if (current_menu_ == &main_menu_) return;

  Submenu* submenu = static_cast<Submenu*>(current_menu_);
  current_menu_ = submenu->parent();
  MakeCursesMenus();
  post_menu(current_curses_menu_);
}

void MenuApplication::FreeCursesMenus() {
  if (current_curses_items_ != nullptr) {
    for (size_t i = 0; i < current_menu_->size(); i++) {
      free_item((current_curses_items_.get())[i]);
    }
  }
  if (current_curses_menu_ != nullptr) {
    free_menu(current_curses_menu_);
  }
}

void MenuApplication::MakeCursesMenus() {
  FreeCursesMenus();

  ITEM** items_raw = nullptr;
  current_menu_->GetCursesMenu(&current_curses_menu_, &items_raw);

  current_curses_items_.reset(items_raw);
}

}
