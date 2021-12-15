HTTP_202_ACCEPTED = {
    202: {
        'description': 'Successful Response',
        'content': {
            'application/json': {
                'example': 'string'
            }
        }
    }
}

HTTP_400_BAD_REQUEST = {
    400: {
        'description': 'Bad request',
        'content': {
            'application/json': {
                'example': {
                    'detail': 'string'
                }
            }
        }
    }
}

HTTP_401_UNAUTHORIZED = {
    401: {
        'description': 'Unauthorized',
        'content': {
            'application/json': {
                'example': {
                    'detail': 'string'
                }
            }
        }
    }
}

HTTP_500_INTERNAL_SERVER_ERROR = {
    500: {
        'description': 'Internal Server Error',
        'content': {
            'application/json': {
                'example': {
                    'detail': ['string']
                }
            }
        }
    }
}

HTTP_503_SERVICE_UNAVAILABLE = {
    503: {
        'description': 'Service Unavailable',
        'content': {
            'application/json': {
                'example': {
                    'detail': 'string'
                }
            }
        }
    }
}
