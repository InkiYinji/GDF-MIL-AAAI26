

class RecordWriter:

    def __init__(self, file_path, mode="a"):
        self.file_path = file_path
        self.mode = mode

    def write2file(self, context, is_print=True, end="\n"):
        with open(self.file_path, self.mode) as log_txt:
            if is_print:
                print(context)
            log_txt.write(context + end)
        log_txt.close()
