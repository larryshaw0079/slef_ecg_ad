import numpy as np
import torch
from tqdm.std import tqdm


def evaluate(model, args, writer, transformation, train_loader, test_loader, test_mode='test', score_mode='dirichlet'):
    ############################################################
    #                        Evaluation                        #
    ############################################################
    model.eval()

    truth = []
    score = []

    if score_mode == 'simple':
        predictions = []
        for sample, label in tqdm(test_loader):
            batch_y_hat = np.zeros((sample.size(0), args.num_trans))
            for t in range(args.num_trans):
                x = transformation.apply_transformation_single(sample, t)
                x = x.cuda()

                with torch.no_grad():
                    batch_y_hat[:, t] = torch.softmax(model(x), dim=1).cpu().numpy()[:, t]

            predictions.append(batch_y_hat)
            truth.append(label.numpy())

        predictions = np.concatenate(predictions)

        score = simplified_normality_score(predictions)
        truth = np.concatenate(truth).reshape(-1)
    elif score_mode == 'dirichlet':
        score = None

        for t in tqdm(range(args.num_trans)):
            observations = []
            predictions = []

            for sample in train_loader:
                x = transformation.apply_transformation_single(sample, t)
                x = x.cuda()

                with torch.no_grad():
                    batch_observations = torch.softmax(model(x), dim=1).cpu().numpy()

                observations.append(batch_observations)

            observations = np.concatenate(observations)

            for sample, label in test_loader:
                x = transformation.apply_transformation_single(sample, t)
                x = x.cuda()

                with torch.no_grad():
                    batch_predictions = torch.softmax(model(x), dim=1).cpu().numpy()

                predictions.append(batch_predictions)
                if t == 0:
                    truth.append(label.numpy())

            predictions = np.concatenate(predictions)

            if score is None:
                score = np.zeros((predictions.shape[0]))
            score += normality_score(observations, predictions)
        score /= args.num_trans
        truth = np.concatenate(truth).reshape(-1)
    else:
        raise ValueError('Invalid score mode!')

    if test_mode == 'val':
        performance, roc_fig, pr_fig = get_performance(score, truth)

        return performance
    elif test_mode == 'test':
        performance, roc_fig, pr_fig = get_performance(score, truth, plot=True)

        return performance, roc_fig, pr_fig
    else:
        raise ValueError('Invalid evaluation mode!')
