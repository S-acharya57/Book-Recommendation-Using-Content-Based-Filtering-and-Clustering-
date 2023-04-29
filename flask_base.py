from flask import Flask, render_template, url_for, redirect, request
import author_rating_combo, content_based_filtering, cluster_try, collaborative
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    login_required,
    logout_user,
    current_user,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(
    "dataset_cluster_added.xlsx",
)
df.fillna(value="", inplace=True)


df.drop_duplicates(subset=["book_title"], inplace=True)


df["genres"] = df["genres"].apply(lambda x: x.split("|"))

df["book_authors"] = df["book_authors"].apply(lambda x: x.split("|"))

df["book_pages"] = pd.to_numeric(
    df["book_pages"].str.replace(" pages", ""), errors="coerce"
)

df["book_pages"] = (
    df["book_pages"]
    .astype(str)
    .str.replace(" pages", "")
    .replace("", np.nan)
    .astype(float)
)


@app.route("/recommend_collaborative", methods=["GET", "POST"])
def recommend_collaborative():
    if request.method == "POST":
        # get form data
        cluster = int(request.form["cluster"])
        user_id = int(request.form["user_id"])
        # title = request.form["title"]
        # print(title, cluster, df.head)
        books = collaborative.recommend(user_id, cluster)
        top = books
        # print(top)
        return render_template(
            "recommend_collaborative.html",
            book_name=list(top["book_title"].values),
            author=list(top["book_authors"].values),
            image=list(top["image_url"].values),
            votes=list(top["book_rating_count"].values),
            rating=list(top["book_rating"].values),
            indices=list(top.index),
        )
    else:
        return render_template("recommend_collaborative.html")


@app.route("/recommend_genre")
def recommend_genre():
    if request.method == "POST":
        # get form data
        title = request.form["title"]
        print(title)
        books = cluster_try.recommend(title)
        top = books
        print(top)
        return render_template(
            "recommend.html",
            book_name=list(top["book_title"].values),
            author=list(top["book_authors"].values),
            image=list(top["image_url"].values),
            votes=list(top["book_rating_count"].values),
            rating=list(top["book_rating"].values),
            indices=list(top.index),
        )
    else:
        return render_template("recommend.html")


@app.route("/")
def index():
    top = df.head(30)
    return render_template(
        "index.html",
        book_name=list(top["book_title"].values),
        author=list(top["book_authors"].values),
        image=list(top["image_url"].values),
        votes=list(top["book_rating_count"].values),
        rating=list(top["book_rating"].values),
        indices=list(top.index),
    )


@app.route("/book/<int:index>")
def book_desc(index):
    book = df.iloc[index]
    return render_template("book_description.html", book=book)


@app.route("/recommend_desc", methods=["GET", "POST"])
def recommend_desc():
    if request.method == "POST":
        # get form data
        cluster = int(request.form["cluster"])
        title = request.form["title"]
        # print(title, cluster, df.head)
        books = content_based_filtering.recommend(cluster, title, df)
        top = books
        # print(top)
        return render_template(
            "recommend_desc.html",
            book_name=list(top["book_title"].values),
            author=list(top["book_authors"].values),
            image=list(top["image_url"].values),
            votes=list(top["book_rating_count"].values),
            rating=list(top["book_rating"].values),
            indices=list(top.index),
        )
    else:
        return render_template("recommend_desc.html")


@app.route("/recommend_author_rating", methods=["GET", "POST"])
def recommend_author_rating():
    if request.method == "POST":
        # get form data
        title = request.form["title"]
        print(title)
        books = author_rating_combo.recommend(title, df)
        top = books
        print(top)
        return render_template(
            "recommend.html",
            book_name=list(top["book_title"].values),
            author=list(top["book_authors"].values),
            image=list(top["image_url"].values),
            votes=list(top["book_rating_count"].values),
            rating=list(top["book_rating"].values),
            indices=list(top.index),
        )
    else:
        return render_template("recommend.html")


@app.route("/recommend_books", methods=["post"])
def recommend():
    user_input = request.form.get("user_input")
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True
    )[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books["Book-Title"] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values))

        data.append(item)

    print(data)

    return render_template("recommend.html", data=data)


"""
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), Length(min=4, max=20)],
        render_kw={"placeholder": "Username"},
    )

    password = PasswordField(
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "Password"},
    )

    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                "That username already exists. Please choose a different one."
            )


class LoginForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), Length(min=4, max=20)],
        render_kw={"placeholder": "Username"},
    )

    password = PasswordField(
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "Password"},
    )

    submit = SubmitField("Login")


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for("dashboard"))
    return render_template("login.html", form=form)


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html", form=form)

"""
if __name__ == "__main__":
    app.run(debug=True)
