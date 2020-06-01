from EditingMethodsClasses import EditingMethods
from VersionControl import ImageVersionController
from HybridAlgorithmImplementation import HybridAlgorithm


class ImageEditingController:
    def __init__(self, image, containtment_dict):
        self.image = image
        self.methods = EditingMethods()
        self.verse = ImageVersionController(self.image)
        self.containtment = []
        # for item in containtment_dict:
        #     if item[0] = 'param':
        #         self.containtment.append(['param'])

        self.actions = self.methods.operations + self.verse.operations + ['break', 'show', 'save']

    def editing_process(self):
        command = ''
        while command != 'break':
            command = input('Enter command:')
            if command in self.actions:
                image = self.verse.get_current_image()

                if command in self.methods.operations:
                    if command == 'blur':
                        image = self.methods.blurred(image, int(input('Enter blurring degree:')))

                    if command == 'sharpen':
                        image = self.methods.sharped(image, int(input('Enter sharpen degree:')))

                    image.show()
                    self.verse.ask_bout_changes(image)

                if command in self.verse.operations:
                    if command == 'undo':
                        print(self.verse.undo())
                    if command == 'redo':
                        print(self.verse.redo())
                    if command == 'initial':
                        self.verse.get_initial().show()

                if command == 'show':
                    self.verse.get_current_image().show()

                if command == 'save':
                    image.save(input('Enter path to image:'))

            else:
                print('Wrong command! Try again.')

        return self.verse.get_current_image()
