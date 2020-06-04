from EditingMethodsClasses import EditingMethods
from VersionControl import ImageVersionController
from LayersControl import ImageLayersController
import os


class ImageEditingController:
    def __init__(self, image, log=False, log_path=os.getcwd(), save=False, save_path=os.getcwd()):
        self.image = image

        self.methods = EditingMethods()
        self.layers = ImageLayersController(self.image, layers_logged=log, log_path=log_path, layers_extracted=False)
        self.verse = ImageVersionController(self.image, self.layers.lab)


        # self.containment = []
        # for item in containment:
        #     if item[0] = 'param':
        #         self.containment.append(['param'])

        self.actions = self.methods.operations + self.verse.operations \
                       + self.layers.operations + ['break', 'show', 'save']

    def editing_process(self):
        command = ''
        working_with_layer = False
        branch_list = []
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

                if command in self.layers.operations:
                    if command == 'get_layers':
                        line = input('Enter layers numbers through comma like 1, 2, 3:')
                        self.layers.temp_merge_few([int(x) for x in line.split(", ")]).show()

                    if command == 'merge_layers':
                        line = input('Enter layers numbers through comma like 1, 2, 3:')

                        self.verse.ask_bout_changes(self.layers.const_merge_few([int(x) for x in line.split(", ")]))

                    if command == 'return all':
                        self.layers.return_all()

                    if command == 'show_layers':
                        self.layers.show_different_layers()

                    if command == 'work_with_layer':
                        if not working_with_layer:
                            line = input('Enter layers numbers through comma like 1, 2, 3:')
                            branch_list = [int(x) for x in line.split(", ")]
                            self.verse.pull_layer(self.layers.temp_merge_few(branch_list))
                            working_with_layer = True
                        else:
                            print('Error! Current image is full.')

                    if command == 'return_to_whole':
                        if working_with_layer:
                            self.verse.push_layer(self.layers.merge_pictures(self.verse.current_image,\
                                                                             self.verse.image_stack[-1], branch_list))
                            working_with_layer = False
                        else:
                            print('Error! Current image is full.')

                if command == 'show':
                    self.verse.get_current_image().show()

                if command == 'save':
                    image.save(input('Enter path to image:'))
            else:
                print('Wrong command! Try again.')

        return self.verse.get_current_image()
# aaaaaaaaaaaaaaa