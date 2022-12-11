#!/usr/bin/python

'''Self signed x509 certificate generator

can look at generated file using openssl:
    openssl x509 -inform pem -in selfsigned.crt -noout -text'''
import sys

from OpenSSL import crypto, SSL

from skynet_bot.constants import DEFAULT_CERTS_DIR


def input_or_skip(txt, default):
    i = input(f'[default: {default}]: {txt}')
    if len(i) == 0:
        return default
    else:
        return i


if __name__ == '__main__':
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = input('country name two char ISO code (example: US): ')
    cert.get_subject().ST = input('state or province name (example: Texas): ')
    cert.get_subject().L = input('locality name (example: Dallas): ')
    cert.get_subject().O = input('organization name: ')
    cert.get_subject().OU = input_or_skip('organizational unit name: ', 'none')
    cert.get_subject().CN = input('common name: ')
    cert.get_subject().emailAddress = input('email address: ')
    cert.set_serial_number(int(input_or_skip('numberic serial number: ', 0)))
    cert.gmtime_adj_notBefore(int(input_or_skip('amount of seconds until cert is valid: ', 0)))
    cert.gmtime_adj_notAfter(int(input_or_skip('amount of seconds until cert expires: ', 10*365*24*60*60)))
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha512')
    with open(f'{DEFAULT_CERTS_DIR}/{sys.argv[1]}.cert', "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
    with open(f'{DEFAULT_CERTS_DIR}/{sys.argv[1]}.key', "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))
