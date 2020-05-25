class ImageVersionController:
    def __init__(self, image):
        self.image_stack = [image]
        self.current_image = self.image_stack[0]
        self.operations = ['undo', 'redo', 'initial']
        self.current_version_id = 0

    def confirm_changes(self, image):
        # if current_image != the last in the stack
        if self.current_version_id < len(self.image_stack) - 1:
            # resize the stack to current size by deleting next values
            del self.image_stack[self.current_version_id:]
        # appending image to stack and declaring it a current version
        self.image_stack.append(image)
        self.current_image = image
        self.current_version_id += 1

    def abort_changes(self):
        self.current_image = self.image_stack[-1]

    def get_current_image(self):
        return self.current_image

    def get_initial(self):
        return self.image_stack[0]

    def ask_bout_changes(self, image):
        print('Save changes? Y/N:')
        answer_str = input()
        while answer_str not in ['Y', 'N']:
            print('Wrong command, try again')
            answer_str = input()
        if answer_str == 'Y':
            self.confirm_changes(image)
        else:
            self.abort_changes()

    def undo(self):
        if self.current_version_id > 0:
            self.current_version_id -= 1
            self.current_image = self.image_stack[self.current_version_id]
            return 'change undone!'
        else:
            return 'current version is the initial one!'

    def redo(self):
        if self.current_version_id < len(self.image_stack) - 1:
            self.current_version_id += 1
            self.current_image = self.image_stack[self.current_version_id]
            return 'change redone'
        else:
            return 'current version is the last one!'

    def return_to_initial(self):
        self.image_stack = [self.image_stack[0]]
        self.current_image = self.image_stack[0]
        self.current_version_id = 0
