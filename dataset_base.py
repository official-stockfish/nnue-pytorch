class TrainingDataProvider:
    def __init__(self, create_stream, destroy_stream, get_next_part, destroy_part, filename, batch_size=None):
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.get_next_part = get_next_part
        self.destroy_part = destroy_part
        self.batch_size = batch_size
        self.filename = filename
        self.stream = self.create_stream(filename.encode('utf-8'))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size:
            v = self.get_next_part(self.stream, self.batch_size)
        else:
            v = self.get_next_part(self.stream)

        if v:
            tensors = v.contents.get_tensors()
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)
