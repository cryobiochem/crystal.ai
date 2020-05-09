from webapp import db

class user(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.Text)
    email = db.Column(db.Text)
    password = db.Column(db.Text)
    created_on = db.Column(db.DateTime)
    last_change = db.Column(db.DateTime)

    def __init__(self,username,email,password,created_on,last_change):
        self.username = username
        self.email = email
        self.password = password
        self.created_on = created_on
        self.last_change = last_change

    def __repr__(self):
        return self.id,self.username,self.email,self.created_on,self.last_change
