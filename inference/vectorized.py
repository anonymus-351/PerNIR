import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm


def as_sparse_vector(basket, num_items):
    column_indexes = np.zeros(len(basket))
    row_indexes = basket
    values = np.ones(len(basket))
    return coo_matrix((values, (row_indexes, column_indexes)), shape=(num_items, 1))


def history_vector(B_u, num_items):
    h_u = coo_matrix(([], ([], [])), shape=(num_items, 1))
    for t in range(len(B_u)):
        h_u += B_u[t] * (1.0 / (len(B_u) - t))
    return h_u


def basket_cooccurrence_matrix(B_u, num_items):
    C_u = coo_matrix(([], ([], [])), shape=(num_items, num_items))

    for t in range(len(B_u)):

        items = B_u[t].nonzero()[0]
        tmp_row_indexes = []
        tmp_column_indexes = []
        tmp_values = []

        for index_a in range(len(items)):
            item_a = items[index_a]
            for index_b in range(len(items)):
                item_b = items[index_b]
                if index_a < index_b:
                    weighted_distance = 1.0 / ((len(B_u) - t) * abs(index_a - index_b))
                    tmp_row_indexes.append(item_a)
                    tmp_column_indexes.append(item_b)
                    tmp_values.append(weighted_distance)
                    # Add symmetric entry as well
                    tmp_row_indexes.append(item_b)
                    tmp_column_indexes.append(item_a)
                    tmp_values.append(weighted_distance)

        C_u += coo_matrix((tmp_values, (tmp_row_indexes, tmp_column_indexes)), shape=(num_items, num_items))

    return C_u


def selection_vector(incomplete_basket, num_items):
    column_indexes = np.zeros(len(incomplete_basket))
    row_indexes = incomplete_basket
    values = [1.0 / (len(incomplete_basket) - i) for i in range(len(incomplete_basket))]

    return coo_matrix((values, (row_indexes, column_indexes)), shape=(num_items, 1))


def history_and_coocc_from_neighbors(h_u, C_u, baskets_of_similar_users, num_items):

    h_N_u = coo_matrix(([], ([], [])), shape=(num_items, 1))
    C_N_u = coo_matrix(([], ([], [])), shape=(num_items, num_items))

    for baskets in baskets_of_similar_users:

        B_v = [as_sparse_vector(basket, num_items) for basket in baskets]
        h_v = history_vector(B_v, num_items)

        similarity = ((h_u.T * h_v) / (norm(h_u) * norm(h_v))).data[0]

        h_N_u += similarity * h_v
        C_N_u += similarity * C_u

    num_similar_users = len(baskets_of_similar_users)
    h_N_u /= num_similar_users
    C_N_u /= num_similar_users

    return h_N_u, C_N_u
