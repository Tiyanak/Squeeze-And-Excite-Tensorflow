import numpy as np
from utils import constant, util
import tensorflow as tf
import time
import model
from TFRecordsReader import TFRecordsReader

class CNN_Records():

    def __init__(self):

        with tf.device('/device:GPU:1'):
            self.model = model.Model()
            self.tfTrainReader = TFRecordsReader()
            self.tfTrainEvalReader = TFRecordsReader()
            self.tfValidEvalReader = TFRecordsReader()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        self.sess = tf.Session(config=config)

    def train(self):

        self.sess.as_default()

        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['lr'] = []

        num_examples = 45000
        max_epochs = constant.config['max_epochs']
        batch_size = constant.config['batch_size']
        num_batches = num_examples // batch_size
        log_every = constant.config['log_every']
        mean_time = []

        self.tfTrainReader.create_iterator("train", max_epochs, batch_size, num_batches * batch_size)
        self.tfTrainEvalReader.create_iterator("train", 1, batch_size, num_batches * batch_size)
        self.tfValidEvalReader.create_iterator("valid", 1, batch_size, num_batches * batch_size)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        for epoch_num in range(1, max_epochs + 1):

            for step in range(num_batches):

                batch_x, batch_y = self.sess.run([self.tfTrainReader.images, self.tfTrainReader.labels])
                batch_y = util.class_to_onehot(batch_y, constant.config['num_class'])

                feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y}
                start_time = time.time()

                run_ops = [self.model.train_op, self.model.loss, self.model.logits]
                ret_val = self.sess.run(run_ops, feed_dict=feed_dict)
                _, loss_val, logits_val = ret_val

                duration = time.time() - start_time
                mean_time.append(duration)

                if (step + 1) * batch_size % log_every == 0:
                    sec_per_batch = float(duration)
                    self.log_step(epoch_num, step, batch_size, num_batches * batch_size, loss_val, sec_per_batch)

            print("EPOCH STATISTICS : ")

            train_loss, train_acc = self.validate(epoch_num, self.tfTrainEvalReader, "train")
            valid_loss, valid_acc = self.validate(epoch_num, self.tfValidEvalReader, "valid")
            print("Total epoch time training: {}".format(np.mean(mean_time)))

            mean_time.clear()

            lr = self.sess.run([self.model.learning_rate])

            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [valid_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [valid_acc]
            plot_data['lr'] += [lr]

            util.plot_training_progress(constant.PLOT_TRAINING_SAVE_DIR, plot_data)

        coord.request_stop()

        coord.join(threads)
        self.sess.close()

    def predict(self, X):

        preds = self.sess.run(self.model.prediction, feed_dict={self.model.X: X})

        return preds

    def log_step(self, epoch, step_batch, batch_size, total_batches, loss, sec_per_batch):

        format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
        print(format_str % (epoch, (step_batch + 1) * batch_size, total_batches, loss, sec_per_batch))

    def validate(self, epoch, reader, dataset_type="Unknown"):

        num_examples = 45000 if dataset_type == 'train' else 5000
        batch_size = constant.config['batch_size']
        num_batches = num_examples // batch_size

        losses = []
        eval_preds = []

        labels = np.ndarray((0,), dtype=np.int64)

        for step in range(num_batches):

            batch_x, batch_y = self.sess.run([reader.images, reader.labels])
            labels =  np.concatenate((labels, batch_y), axis=0)

            feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y}
            run_ops = [self.model.loss, self.model.prediction]

            ret_val = self.sess.run(run_ops, feed_dict=feed_dict)
            loss, preds = ret_val

            losses.append(loss)
            eval_preds.append(preds)

        eval_preds = np.vstack(eval_preds)
        total_loss = np.mean(losses)

        labels = util.class_to_onehot(labels, constant.config['num_class'])

        acc, pr = util.eval_perf_multi(np.argmax(labels, axis=1), np.argmax(eval_preds, axis=1))
        print("{} error: epoch {} loss={} accuracy={}".format(dataset_type, epoch, total_loss, acc))

        return total_loss, acc