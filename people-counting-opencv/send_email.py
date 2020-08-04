import smtplib
from email.mime.text import MIMEText

def send_Email(msge):

	msg = MIMEText(msge)

	sender = 'maskdetector101@gmail.com'
	you = 'srinivassriram06@gmail.com'
	msg['Subject'] = 'Test email'
	msg['From'] = me
	msg['To'] = you

	s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	s.login("maskdetector101@gmail.com", "LearnIOT06!")
	s.send_message(msg)
	s.quit()


number = str(45)

send_Email(number)
