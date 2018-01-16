import numpy as np
from utils import constant, util
import tensorflow as tf
import time
import model

class CNN():

    def __init__(self):

        with tf.device('/device:GPU:0'):
            self.model = model.Model()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        self.sess = tf.Session(config=config)

    def train(self, dataset):

        self.sess.run(tf.global_variables_initializer())

        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['lr'] = []

        num_examples = dataset.train_x.shape[0]
        max_epochs = constant.config['max_epochs']
        batch_size = constant.config['batch_size']
        num_batches = num_examples // batch_size
        log_every = constant.config['log_every']
        mean_time = []

        for epoch_num in range(1, max_epochs + 1):

            dataset.train_x, dataset.train_y = util.shuffle_data(dataset.train_x, dataset.train_y)

            for step in range(num_batches):

                offset = step * batch_size

                batch_x = dataset.train_x[offset:(offset + batch_size), ...]
                batch_y = dataset.train_y[offset:(offset + batch_size)]

                feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y}
                start_time = time.time()

                run_ops = [self.model.train_op, self.model.loss, self.model.logits] # +self.weights
                ret_val = self.sess.run(run_ops, feed_dict=feed_dict)
                _, loss_val, logits_val = ret_val # +weights

                duration = time.time() - start_time
                mean_time.append(duration)

                if (step + 1) * batch_size % log_every == 0:
                    sec_per_batch = float(duration)
                    self.log_step(epoch_num, step, batch_size, num_batches * batch_size, loss_val, sec_per_batch)

            print("EPOCH STATISTICS : ")

            train_loss, train_acc = self.validate(epoch_num, dataset.train_x, dataset.train_y, "Train")
            valid_loss, valid_acc = self.validate(epoch_num, dataset.valid_x, dataset.valid_y, "Validation")
            print("Total epoch time training: {}".format(np.mean(mean_time)))

            mean_time.clear()

            lr = self.sess.run([self.model.learning_rate])

            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [valid_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [valid_acc]
            plot_data['lr'] += [lr]

        util.plot_training_progress(plot_data)

    def predict(self, X):

        preds = self.sess.run(self.model.prediction, feed_dict={self.model.X: X})

        return preds

    def log_step(self, epoch, step_batch, batch_size, total_batches, loss, sec_per_batch):

        format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
        print(format_str % (epoch, (step_batch + 1) * batch_size, total_batches, loss, sec_per_batch))

    def validate(self, epoch, inputs, labels, dataset_type="Unknown"):

        num_examples = inputs.shape[0]
        batch_size = constant.config['batch_size']
        num_batches = num_examples // batch_size

        losses = []
        eval_preds = []

        for step in range(num_batches):

            offset = step * batch_size

            batch_x = inputs[offset:(offset + batch_size), ...]
            batch_y = labels[offset:(offset + batch_size)]

            feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y}
            run_ops = [self.model.loss, self.model.prediction]

            ret_val = self.sess.run(run_ops, feed_dict=feed_dict)
            loss, preds = ret_val

            losses.append(loss)
            eval_preds.append(preds)

        eval_preds = np.vstack(eval_preds)
        total_loss = np.mean(losses)

        acc, pr = util.eval_perf_multi(np.argmax(labels, axis=1), np.argmax(eval_preds, axis=1))
        print("{} error: epoch {} loss={} accuracy={}".format(dataset_type, epoch, total_loss, acc))

        return loss, acc