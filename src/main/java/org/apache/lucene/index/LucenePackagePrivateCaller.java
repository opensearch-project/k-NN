/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.security.AccessController;
import java.security.PrivilegedAction;

@Log4j2
public class LucenePackagePrivateCaller {

    public static Object callPrivateFieldWithMethod(Class<?> clz, String fieldName, String methodName, Object called) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                Field field = clz.getDeclaredField(fieldName);
                field.setAccessible(true);
                return callMethod(field.getType(), methodName, null, field.get(called), null);
            } catch (Exception e) {
                log.error("callPrivateFieldWithMethod", e);
                throw new RuntimeException(e);
            }
        });
    }

    public static Object callMethod(Class<?> clz, String methodName, Class<?>[] argTypes, Object called, Object[] args) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                Method method = clz.getDeclaredMethod(methodName, argTypes);
                return callMethod(method, called, args);
            } catch (Exception e) {
                log.error("callMethod", e);
                throw new RuntimeException(e);
            }
        });
    }

    public static Object callMethod(Method method, Object called, Object... args) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                method.setAccessible(true);
                return method.invoke(called, args);
            } catch (Exception e) {
                log.error("callMethod", e);
                throw new RuntimeException(e);
            }
        });
    }
}
