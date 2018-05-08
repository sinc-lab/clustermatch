import tempfile


def get_temp_file_name(file_extension=''):
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmpfile:
        temp_file_name = tmpfile.name
    return temp_file_name
