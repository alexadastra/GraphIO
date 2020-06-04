class ImageVersionController:
    def __init__(self, image, matrix_):
        # stacks and current values
        self.image_stack = [image]
        self.layer_stack = [matrix_]

        self.current_image = self.image_stack[0]
        self.current_layer = self.layer_stack[0]

        self.branch_stack = []
        self.branch_active = False
        self.branch_version_id = 0
        # class operations
        self.operations = ['undo', 'redo', 'initial', 'pull_layer', 'push_layer']
        self.current_version_id = 0

    # updating some of stacks, depending on input
    def confirm_changes(self, image, matr):
        if self.branch_active:  # inserting to branch stack
            if self.branch_version_id < len(self.branch_stack) - 1:
                del self.branch_stack[self.branch_version_id]
            self.branch_stack.append(image)
            self.current_image = image
            self.branch_version_id += 1

        elif matr is None and image is not None:  # inserting to image stack
            # if current_image != the last in the stack
            if self.current_version_id < len(self.image_stack) - 1:
                # resize the stack to current size by deleting next values
                del self.image_stack[self.current_version_id:]
            # appending image to stack and declaring it a current version
            self.image_stack.append(image)
            self.layer_stack.append(self.layer_stack[-1])
            self.current_image = image
            self.current_version_id += 1
        elif matr is not None and image is None:  # inserting to matrix stack
            # if current_image != the last in the stack
            if self.current_version_id < len(self.layer_stack) - 1:
                # resize the stack to current size by deleting next values
                del self.layer_stack[self.current_version_id:]
            # appending image to stack and declaring it a current version
            self.image_stack.append(self.image_stack[-1])
            self.layer_stack.append(matr)
            self.current_layer = matr
            self.current_version_id += 1

    def abort_changes(self):  # set current value to the last one
        if self.branch_active:
            self.current_image = self.branch_stack[-1]
        else:
            self.current_image = self.image_stack[-1]
            self.current_layer = self.layer_stack[-1]

    # image getter
    def get_current_image(self):
        return self.current_image

    # matrix getter
    def get_current_lab(self):
        return self.current_layer

    # initial image getter
    def get_initial_image(self):
        if self.branch_active:
            return self.branch_stack[0]
        else:
            return self.image_stack[0]

    # initial matrix getter
    def get_initial_matr(self):
        return self.layer_stack[0]

    # confirming changes
    def ask_bout_changes(self, image=None, matr=None):
        print('Save changes? Y/N:')
        answer_str = input()
        while answer_str not in ['Y', 'N']:
            print('Wrong command, try again')
            answer_str = input()
        if answer_str == 'Y':
            self.confirm_changes(image, matr)
        else:
            self.abort_changes()

    def undo(self):
        if self.branch_active:
            if self.branch_version_id > 0:
                self.branch_version_id -= 1
                self.current_image = self.branch_stack[self.branch_version_id]
                return 'change undone!'
            else:  # warn that image is the first
                return 'current version is the initial one!'
        else:
            if self.current_version_id > 0:
                self.current_version_id -= 1
                self.current_image = self.image_stack[self.current_version_id]
                self.current_layer = self.layer_stack[self.current_version_id]
                return 'change undone!'
            else:  # warn that image is the first
                return 'current version is the initial one!'

    def redo(self):
        if self.branch_active:
            if self.branch_version_id < len(self.branch_stack) - 1:
                self.current_version_id += 1
                self.branch_version_id = self.image_stack[self.branch_version_id]
                return 'change redone'
            else:  # warn that image is the last
                return 'current version is the last one!'

        else:
            if self.current_version_id < len(self.image_stack) - 1:
                self.current_version_id += 1
                self.current_image = self.image_stack[self.current_version_id]
                self.current_layer = self.layer_stack[self.current_version_id]
                return 'change redone'
            else:  # warn that image is the last
                return 'current version is the last one!'

    # redefining to stack beginning
    def return_to_initial(self):
        if self.branch_active:
            self.branch_stack = [self.branch_stack[0]]
            self.current_image = self.branch_stack[0]
        else:
            self.image_stack = [self.image_stack[0]]
            self.current_image = self.image_stack[0]

            self.layer_stack = [self.layer_stack[0]]
            self.current_layer = self.current_layer[0]

            self.current_version_id = 0

    # starting branch
    def pull_layer(self, image):
        if not self.branch_active:
            self.branch_active = True
            self.branch_stack = [image]
            self.current_image = self.branch_stack[0]

    # ending branch
    def push_layer(self, image):
        if self.branch_active:
            self.branch_active = False
            self.confirm_changes(image=image, matr=None)
            self.branch_stack = []
            self.branch_version_id = 0
