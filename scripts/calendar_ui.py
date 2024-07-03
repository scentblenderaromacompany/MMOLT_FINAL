import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QCalendarWidget, QLabel, QPushButton, QTimeEdit, QLineEdit, QFormLayout, QDialog)
from PyQt5.QtCore import QDate, QTime

class CalendarUI(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Calendar Scheduler')
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()
        
        # Calendar Widget
        self.calendar = QCalendarWidget(self)
        self.calendar.setGridVisible(True)
        self.layout.addWidget(self.calendar)
        
        # Time Edit
        self.time_edit = QTimeEdit(self)
        self.time_edit.setTime(QTime.currentTime())
        self.layout.addWidget(self.time_edit)

        # Input Fields for Marketplace, Item, and Note
        form_layout = QFormLayout()
        self.marketplace_input = QLineEdit(self)
        self.item_input = QLineEdit(self)
        self.note_input = QLineEdit(self)
        form_layout.addRow("Marketplace:", self.marketplace_input)
        form_layout.addRow("Item:", self.item_input)
        form_layout.addRow("Note:", self.note_input)
        self.layout.addLayout(form_layout)
        
        # Schedule Button
        self.schedule_button = QPushButton('Schedule Listing', self)
        self.schedule_button.clicked.connect(self.schedule_listing)
        self.layout.addWidget(self.schedule_button)
        
        # Set Layout
        self.setLayout(self.layout)

    def schedule_listing(self):
        selected_date = self.calendar.selectedDate()
        selected_time = self.time_edit.time()
        datetime_str = f"{selected_date.toString()} {selected_time.toString()}"
        marketplace = self.marketplace_input.text()
        item = self.item_input.text()
        note = self.note_input.text()
        
        # Here, you can add the logic to handle the scheduled listing
        # For example, save the data to a database or a file
        print(f"Scheduled: {datetime_str}, Marketplace: {marketplace}, Item: {item}, Note: {note}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalendarUI()
    window.show()
    sys.exit(app.exec_())
