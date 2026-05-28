from auth_config import USERNAME, PASSWORD


def login(username, password):
    return (
        username == USERNAME
        and password == PASSWORD
    )